"""OrTools utilities."""
import concurrent.futures
import random
from datetime import date, datetime, timedelta
from time import process_time

import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2  # noqa: F401
from tqdm import tqdm

from src.setup import logger


def create_data_model(
    params, distance_matrix=None, durations_matrix=None, time_windows=None
):
    """Prepare data for OrTools."""
    assert (
        distance_matrix is not None or durations_matrix is not None
    ), "At least one between distance and duration matrices should exist"
    num_locations = (
        len(distance_matrix) if distance_matrix is not None else len(durations_matrix)
    )
    data = {
        "depot": 0,  # Start and destination node for all the tours
        "distance_matrix": distance_matrix,
        "time_matrix": durations_matrix,
        "num_locations": num_locations,
        "num_vehicles": params["num_vehicles"],
        "service_time": params["service_time"],
        "demands": params["demands"],
        "pickups": params["pickups"],
        "vehicle_capacities": [params["max_legs"]] * params["num_vehicles"],
        "time_windows": time_windows,
    }
    return data


def register_distance(params, manager, routing, data_ort):
    """Register distance callback."""
    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Return distance between two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data_ort["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # In this example, the arc cost evaluator is the transit_callback_index,
    # which is the solver's internal reference to the distance callback.
    # Therefore, the cost of travel between any two locations is the distance between
    # them. However, in general the costs can involve other factors as well.
    # You can also define multiple arc cost evaluators that depend on which vehicle is
    # traveling between locations, using  routing.SetArcCostEvaluatorOfVehicle()
    if params["solve_by_distance"]:
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # To solve this problem, you need to create a distance dimension,
    # which computes the cumulative distance traveled by each vehicle along its route.
    # You can then set a cost proportional to the max of the distances along each route.
    # --> Add Distance constraint.
    dimension_name = "Distance"
    routing.AddDimension(
        transit_callback_index,
        0,  # a "slack" is a constant quantity, added at each node: here is 0 (distance)
        params["max_distance_vehicles_km"] * 1000,  # vehicle max travel distance [m]
        True,  # start cumulative distance from zero
        dimension_name,
    )

    # Set a cost coefficient for the maximum route length,
    # so the program minimizes (also/mainly) the length of the longest route.
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    if params["solve_by_distance"]:
        distance_dimension.SetGlobalSpanCostCoefficient(
            params["pen_distance_vehicle_factor"]
        )

    return routing, distance_dimension


def register_duration(params, manager, routing, data_ort):
    """Register duration callback."""
    # Create and register a transit callback.
    def time_callback(from_index, to_index):
        """Return distance between two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return (
            data_ort["time_matrix"][from_node][to_node]
            + data_ort["service_time"][to_node]
        )

    transit_callback_index = routing.RegisterTransitCallback(time_callback)

    # Define cost of each arc.
    if not params["solve_by_distance"]:
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Time constraint.
    time = "Time"
    routing.AddDimension(
        transit_callback_index,
        int(params["slack_time_max_h"] * 3600),  # allow max waiting time
        int(params["max_time_tour_h"] * 3600),  # maximum time per vehicle [min]
        False,  # Don't force time to start cumulate from zero.
        time,
    )
    time_dimension = routing.GetDimensionOrDie(time)

    return routing, time_dimension


def constrain_capacity(manager, routing, data_ort):
    """Add capacity constraints."""
    # In addition to the distance callback, the solver also requires a demand callback,
    # which returns the demand at each location,
    # and a dimension for the capacity constraints.

    # Unlike the distance callback, which takes a pair of locations as inputs,
    # the demand callback only depends on the location (from_node) of the delivery.
    def demand_callback(from_index):
        """Return the demand of the node."""
        # Convert from routing variable's Index to NodeIndex of demand.
        from_node = manager.IndexToNode(from_index)
        return data_ort["demands"][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

    dimension_name = "Capacity"
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # a "slack" is a constant quantity, added at each node: here is 0 (load)
        data_ort["vehicle_capacities"],  # vehicle max capacities (Vector not scalar)
        True,  # start cumulative weight to zero
        dimension_name,
    )

    return routing


def constrain_time_windows(manager, routing, time_dim, data_ort):
    """Add visiting time constraints."""
    # Add time window constraints for each location except depot.
    for location_idx, time_window in enumerate(data_ort["time_windows"]):
        if location_idx == data_ort["depot"]:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dim.CumulVar(index).SetRange(time_window[0], time_window[1])

    # Add time window constraints for each vehicle start node.
    depot_idx = data_ort["depot"]

    for vehicle_id in range(data_ort["num_vehicles"]):
        index = routing.Start(vehicle_id)
        time_dim.CumulVar(index).SetRange(
            data_ort["time_windows"][depot_idx][0],
            data_ort["time_windows"][depot_idx][1],
        )

        routing.AddVariableMinimizedByFinalizer(
            time_dim.CumulVar(routing.Start(vehicle_id))
        )
        routing.AddVariableMinimizedByFinalizer(
            time_dim.CumulVar(routing.End(vehicle_id))
        )
    return routing


def constrain_pickup(params, manager, routing, time_dim, data_ort):
    """Add pickup time constraints."""
    # Add constraint to each pickup rule
    for request in data_ort["pickups"]:

        # Collect location indexes
        delivery_index = manager.NodeToIndex(request[0])
        pickup_index = manager.NodeToIndex(request[1])

        # Create pickup-delivery request
        routing.AddPickupAndDelivery(delivery_index, pickup_index)

        # Impose that pickup should happen after delivery (with a slack)
        routing.solver().Add(
            time_dim.CumulVar(pickup_index)
            >= time_dim.CumulVar(delivery_index)
            + data_ort["service_time"][request[1]]
            + int(3600 * params["min_pickup_delay_h"])
        )
    return routing


def solver(params, routing):
    """Solve optimization problem."""
    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    strategy = (
        random.choice(params["strategies"])
        if params["test_mode"]
        else "PARALLEL_CHEAPEST_INSERTION"  # "AUTOMATIC"
    )
    strategy_label = f"routing_enums_pb2.FirstSolutionStrategy.{strategy}"

    local_search = (
        random.choice(params["local_search"])
        if params["test_mode"]
        else "TABU_SEARCH"  # "AUTOMATIC"
    )
    local_search_label = f"routing_enums_pb2.LocalSearchMetaheuristic.{local_search}"

    duration = (
        random.randrange(params["max_search_time_min"] + 1)
        if params["test_mode"]
        else params["max_search_time_min"]
    )

    search_parameters.first_solution_strategy = eval(strategy_label)
    search_parameters.local_search_metaheuristic = eval(local_search_label)
    search_parameters.time_limit.seconds = int(60 * duration)

    # Solve the problem.
    t_start = process_time()
    solution = routing.SolveWithParameters(search_parameters)
    t_end = process_time()
    opt_duration = t_end - t_start

    return solution, strategy, local_search, opt_duration


def print_solution(
    params, data, manager, routing, solution, strategy, local_search, opt_duration
):
    """Print solution on console and file."""
    # Init constants
    time_dimension = routing.GetDimensionOrDie("Time")

    routing_solution = {
        "max_route_distance": 0,
        "sum_total_distances": 0,
        "max_route_time": 0,
        "sum_total_time": 0,
        "no_vehicles": 0,
        "strategy": strategy,
        "local_search": local_search,
        "duration": opt_duration,
        "solution": [],
    }

    output_df_columns = ["Stop ID", "Stop type", "Action", "ETA", "ETD", "Distance"]
    output_df_base = pd.DataFrame(columns=output_df_columns)
    output_df_list = []

    # Print travel for each vehicle
    for vehicle_id in range(data["num_vehicles"]):

        # Initialize travel
        route_distance = 0
        route_load = 0
        n_stores = 0
        n_customers = 0
        n_pickups = 0
        previous_ETD = None
        next_leg_duration = None
        index = routing.Start(vehicle_id)
        output_df = output_df_base.copy()
        out_str = f"Route for vehicle {routing_solution['no_vehicles']}:\n"

        # Define next leg of the travel
        while not routing.IsEnd(index):

            # Define leg start
            node_index = manager.IndexToNode(index)
            route_leg = {"Stop ID": params["stop_id_map"][node_index]}

            # Define leg content
            if data["demands"][node_index] == -1:
                n_stores += 1
                route_leg["Action"] = "Unload"
                route_leg["Stop type"] = "store"
            elif data["demands"][node_index] == -0.1:
                n_customers += 1
                route_leg["Action"] = "Unload"
                route_leg["Stop type"] = "customer"
            elif data["demands"][node_index] == 0.01:
                n_pickups += 1
                route_leg["Action"] = "Pickup"
                route_leg["Stop type"] = "store"
            else:
                route_leg["Action"] = "Load"
                route_leg["Stop type"] = "depot"

            route_load += data["demands"][node_index]

            time_var = time_dimension.CumulVar(index)
            if output_df.shape[0]:
                route_leg["ETA"] = (
                    datetime.combine(date.min, previous_ETD)
                    + timedelta(seconds=next_leg_duration)
                ).time()
            route_leg["ETD"] = (
                datetime.combine(date.min, params["leave_time"])
                + timedelta(seconds=solution.Min(time_var))
            ).time()

            # Define leg end
            previous_node = manager.IndexToNode(index)
            index = solution.Value(routing.NextVar(index))
            new_node = manager.IndexToNode(index)

            # Update total distance
            route_leg["Distance"] = (
                int(data["distance_matrix"][previous_node][new_node]) / 1000
            )
            route_distance += route_leg["Distance"]

            # Compute next duration
            previous_ETD = route_leg["ETD"]
            next_leg_duration = int(data["time_matrix"][previous_node][new_node])

            # Output
            output_df = pd.concat(
                [output_df, pd.DataFrame([route_leg])], ignore_index=True
            )
            out_str += (
                f"\tNode {node_index} ({route_leg['Stop ID']}): {route_leg['Action']} @ {route_leg['Stop type']} -> "
                f"Current Load ({route_load:.2f}) -> "
                f"ETD {route_leg['ETD']} -> "
                f"Drive {route_leg['Distance']:.3f} km to -->\n"
            )

        # Terminate route
        node_index = manager.IndexToNode(index)
        route_leg = {
            "Stop ID": params["stop_id_map"][node_index],
            "Action": "End journey",
            "Stop type": "depot",
        }

        time_var = time_dimension.CumulVar(index)
        time_min = (
            datetime.combine(date.min, params["leave_time"])
            + timedelta(seconds=solution.Min(time_var))
        ).time()
        route_leg["ETA"] = time_min
        route_load += data["demands"][node_index]

        # Output
        output_df = pd.concat([output_df, pd.DataFrame([route_leg])], ignore_index=True)
        out_str += (
            f"\tNode {node_index} ({route_leg['Stop ID']}): {route_leg['Action']} @ {route_leg['Stop type']} -> "
            f"ETA {route_leg['ETA']} after serving "
            f"{n_stores} stores, {n_customers} customers and {n_pickups} pickups.\n"
        )
        out_str += f"Total route distance: {route_distance:.3f} km.\n"

        route_time = int(solution.Min(time_var) / 60)
        out_str += f"Total route time: {route_time} min\n"

        if route_time > 0:
            output_df_list.append(output_df)
            routing_solution["no_vehicles"] += 1
            if params["verbose"]:
                logger.info(out_str)

        # Update distances
        routing_solution["max_route_distance"] = max(
            route_distance, routing_solution["max_route_distance"]
        )
        routing_solution["sum_total_distances"] += route_distance

        # Update durations
        routing_solution["max_route_time"] = max(
            route_time, routing_solution["max_route_time"]
        )
        routing_solution["sum_total_time"] += route_time

    if params["verbose"]:
        logger.info(
            f"Total number of vehicles used: {routing_solution['no_vehicles']}."
        )
        logger.info(
            f"Longest route distance: {routing_solution['max_route_distance']:.3f} km."
        )
        logger.info(
            f"Sum of the route distances: {routing_solution['sum_total_distances']:.3f} km."
        )
        logger.info(f"Longest route time: {routing_solution['max_route_time']} min.")
        logger.info(
            f"Sum of the route durations: {routing_solution['sum_total_time']} min."
        )

    routing_solution["solution"] = output_df_list

    return routing_solution


def solve_vr(params, distances, durations):
    """Solve generic parametrize VRP."""
    if params["verbose"]:
        logger.info("Starting optimization.")

    # Instantiate the data model.
    data_ort = create_data_model(
        params,
        distance_matrix=distances,
        durations_matrix=durations,
        time_windows=params["time_windows"],
    )

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        data_ort["num_locations"], data_ort["num_vehicles"], data_ort["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Impose optimization by either distance or duration
    routing, distance_dim = register_distance(params, manager, routing, data_ort)
    routing, time_dim = register_duration(params, manager, routing, data_ort)

    # Impose visiting time windows constraint
    if params["constrain_tw"]:
        routing = constrain_time_windows(manager, routing, time_dim, data_ort)

    if params["constrain_cap"]:
        routing = constrain_capacity(manager, routing, data_ort)

    if not params["only_delivery"]:
        routing = constrain_pickup(params, manager, routing, time_dim, data_ort)

    # Compute solution
    solution, strategy, local_search, opt_duration = solver(params, routing)

    # Print solution on console.
    if solution:
        routing_solution = print_solution(
            params,
            data_ort,
            manager,
            routing,
            solution,
            strategy,
            local_search,
            opt_duration,
        )
    else:
        routing_solution = {
            "max_route_distance": 0,
            "sum_total_distances": 0,
            "max_route_time": 0,
            "sum_total_time": 0,
            "no_vehicles": 0,
            "strategy": strategy,
            "local_search": local_search,
            "duration": opt_duration,
            "solution": [],
        }
        if params["verbose"]:
            logger.warning("No solution found!")

    return routing_solution


def repeat_tests(params, distances, durations):
    """Execute repeated tests."""
    logger.info(f"Executing {params['num_test']} optimization instances...")
    # Initialize results
    full_tests = []

    with tqdm(total=params["num_test"]) as pbar:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(solve_vr, params, distances, durations)
                for _ in range(params["num_test"])
            }

            for future in concurrent.futures.as_completed(futures):
                full_tests.append(future.result())
                pbar.update()

    # Compute performances
    performances = pd.DataFrame.from_records(
        [
            {k: v for k, v in solution.items() if k != "solution"}
            for solution in full_tests
        ]
    )
    performances.to_csv(f"{params['data_folder']}/{params['output_test_file']}")

    # Return best output
    performances = performances.query("no_vehicles > 0")
    if performances.shape[0]:
        idx_best_solution = (
            performances.sum_total_distances.idxmin()
            if params["solve_by_distance"]
            else performances.sum_total_time.idxmin()
        )
        result = full_tests[idx_best_solution]
    else:
        result = {"solution": []}

    return result
