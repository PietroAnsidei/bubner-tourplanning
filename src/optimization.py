"""OrTools utilities."""
from datetime import date, datetime, timedelta

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

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
            + int(3600 * params["slack_time_max_h"])
        )
    return routing


def solver(params, routing):
    """Solve optimization problem."""
    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
    )
    search_parameters.time_limit.seconds = int(60 * params["max_search_time_min"])

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)
    return solution


def print_solution(params, data, manager, routing, solution):
    """Print solution on console."""
    # Init constants
    time_dimension = routing.GetDimensionOrDie("Time")

    max_route_distance = 0
    sum_total_distances = 0
    max_route_time = 0
    sum_total_time = 0
    vehicle_idx = 0

    # Print travel for each vehicle
    for vehicle_id in range(data["num_vehicles"]):

        # Initialize travel
        route_distance = 0
        route_load = 0
        n_stores = 0
        n_customers = 0
        n_pickups = 0
        index = routing.Start(vehicle_id)
        out_str = f"Route for vehicle {vehicle_idx}:\n"

        # Define next leg of the travel
        while not routing.IsEnd(index):

            # Define leg start
            node_index = manager.IndexToNode(index)
            node_label = params["stop_id_map"][node_index]

            time_var = time_dimension.CumulVar(index)
            time_min = datetime.combine(date.min, params["leave_time"]) + timedelta(
                seconds=solution.Min(time_var)
            )

            route_load += data["demands"][node_index]

            # Define leg content
            if data["demands"][node_index] == -1:
                n_stores += 1
                activity = "Unload @ store"
            elif data["demands"][node_index] == -0.1:
                n_customers += 1
                activity = "Unload @ customer"
            elif data["demands"][node_index] == 0.01:
                n_pickups += 1
                activity = "Pickup @ store"
            else:
                activity = "Load @ depot"

            # Define leg end
            previous_node = manager.IndexToNode(index)
            index = solution.Value(routing.NextVar(index))
            new_node = manager.IndexToNode(index)

            # Update total distance
            leg_distance = int(data["distance_matrix"][previous_node][new_node]) / 1000
            route_distance += leg_distance

            out_str += (
                f"\tNode {node_index} ({node_label}): {activity} -> "
                f"Current Load ({route_load:.2f}) -> "
                f"ETD {time_min.time()} -> "
                f"Drive {leg_distance:.3f} km to -->\n"
            )

        # Terminate route
        node_index = manager.IndexToNode(index)
        node_label = params["stop_id_map"][node_index]

        time_var = time_dimension.CumulVar(index)
        time_min = datetime.combine(date.min, params["leave_time"]) + timedelta(
            seconds=solution.Min(time_var)
        )
        route_load += data["demands"][node_index]

        out_str += (
            f"\tNode {node_index} ({node_label}): End journey @ depot -> "
            f"ETA {time_min.time()} after serving "
            f"{n_stores} stores, {n_customers} customers and {n_pickups} pickups.\n"
        )
        out_str += f"Total route distance: {route_distance:.3f} km.\n"

        route_time = int(solution.Min(time_var) / 60)
        out_str += f"Total route time: {route_time} min\n"

        if route_time > 0:
            logger.info(out_str)
            vehicle_idx += 1

        # Update distances
        max_route_distance = max(route_distance, max_route_distance)
        sum_total_distances += route_distance

        # Update durations
        max_route_time = max(route_time, max_route_time)
        sum_total_time += route_time

    logger.info(f"Total number of vehicles used: {vehicle_idx}.")
    logger.info(f"Longest route distance: {max_route_distance:.3f} km.")
    logger.info(f"Sum of the route distances: {sum_total_distances:.3f} km.")
    logger.info(f"Longest route time: {max_route_time} min.")
    logger.info(f"Sum of the route durations: {sum_total_time} min.")

    return


def solve_vr(params, distances, durations):
    """Solve generic parametrize VRP."""
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
    solution = solver(params, routing)

    # Print solution on console.
    if solution:
        print_solution(params, data_ort, manager, routing, solution)
    else:
        logger.warning("No solution found!")

    return
