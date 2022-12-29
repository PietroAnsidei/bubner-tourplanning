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
    no_points = (
        len(distance_matrix) if distance_matrix is not None else len(durations_matrix)
    )
    data = {
        "distance_matrix": distance_matrix,
        "time_matrix": durations_matrix,
        "demands": [params["max_legs"]] + [-1] * (no_points - 1),
        "num_vehicles": params["num_vehicles"],
        "vehicle_capacities": [params["max_legs"]] * params["num_vehicles"],
        "time_windows": time_windows,
        "depot": 0,  # Start and destination node for all the tours
    }
    return data


def print_solution_vrp(params, data, manager, routing, solution):
    """Print solution on console."""
    # Init constants
    max_route_distance = 0
    sum_distances = 0

    # Log total loss
    # logger.info(f"Overall distance: {solution.ObjectiveValue() / 1000} [km]")

    # Print travel for each vehicle
    for vehicle_id in range(data["num_vehicles"]):

        # Initialize travel
        route_distance = 0
        route_load = 0
        index = routing.Start(vehicle_id)
        out_str = f"Route for vehicle {vehicle_id}:\n"

        # Define next leg of the travel
        while not routing.IsEnd(index):

            # Define leg start
            node_index = manager.IndexToNode(index)
            node_label = params["stop_id_map"][node_index]
            route_load += data["demands"][node_index]

            out_str += f" {node_index} ({node_label}): Load ({route_load}) -> "

            # Define leg end
            previous_index = index
            index = solution.Value(routing.NextVar(index))

            # Update total distance
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )

        # Terminate route
        node_index = manager.IndexToNode(index)
        node_label = params["stop_id_map"][node_index]
        route_load += data["demands"][node_index]

        out_str += f" {node_index} ({node_label}): Load ({route_load})\n"
        out_str += f"Distance of the route: {route_distance / 1000} km.\n"

        logger.info(out_str)

        # Update distances
        max_route_distance = max(route_distance, max_route_distance)
        sum_distances += route_distance

    logger.info(f"Maximum of the route distances: {max_route_distance / 1000} km.")
    logger.info(f"Sum of the route distances: {sum_distances / 1000} km.")

    return


def solve_vrp(params, distances):
    """Solve vehicle routing problem."""
    # Instantiate data for routing problem.
    data_ort = create_data_model(params, distance_matrix=distances)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data_ort["distance_matrix"]), data_ort["num_vehicles"], data_ort["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

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
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(
        params["pen_distance_vehicle_factor"]
    )

    # In addition to the distance callback, the solver also requires a demand callback,
    # which returns the demand at each location,
    # and a dimension for the capacity constraints.
    def demand_callback(from_index):
        """Return the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
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

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print_solution_vrp(params, data_ort, manager, routing, solution)
    else:
        logger.warning("No solution found !")

    return


def print_solution_vrptw(params, data, manager, routing, solution):
    """Print solution on console."""
    # Init constants
    time_dimension = routing.GetDimensionOrDie("Time")
    total_time = 0

    # Print travel for each vehicle
    for vehicle_id in range(data["num_vehicles"]):

        # Initialize travel
        index = routing.Start(vehicle_id)
        out_str = f"Route for vehicle {vehicle_id}:\n"

        # Define next leg of the travel
        while not routing.IsEnd(index):

            # Define leg start
            time_var = time_dimension.CumulVar(index)
            time_min = datetime.combine(date.min, params["leave_time"]) + timedelta(
                seconds=solution.Min(time_var)
            )
            time_max = datetime.combine(date.min, params["leave_time"]) + timedelta(
                seconds=solution.Max(time_var)
            )

            out_str += (
                f" {params['stop_id_map'][manager.IndexToNode(index)]} "
                f"Time({time_min.time()}, {time_max.time()})) -> "
            )

            # Define leg end
            index = solution.Value(routing.NextVar(index))

        # Terminate route
        time_var = time_dimension.CumulVar(index)
        time_min = datetime.combine(date.min, params["leave_time"]) + timedelta(
            seconds=solution.Min(time_var)
        )
        time_max = datetime.combine(date.min, params["leave_time"]) + timedelta(
            seconds=solution.Max(time_var)
        )

        out_str += (
            f" {params['stop_id_map'][manager.IndexToNode(index)]} "
            f"Time({time_min.time()}, {time_max.time()}))\n"
        )
        out_str += f"Time of the route: {int(solution.Min(time_var) / 60)} min\n"
        logger.info(out_str)

        # Update durations
        total_time += solution.Min(time_var)

    print(f"Sum of the route durations: {int(total_time / 60)} min")

    return


def solve_vrptw(params, durations):
    """Solve distance problem."""
    # Instantiate the data problem.
    data_ort = create_data_model(
        params,
        durations_matrix=durations,
        time_windows=params["time_windows"],
    )

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data_ort["time_matrix"]), data_ort["num_vehicles"], data_ort["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def time_callback(from_index, to_index):
        """Return distance between two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data_ort["time_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Time constraint.
    time = "Time"
    routing.AddDimension(
        transit_callback_index,
        params["slack_time_min"] * 60,  # allow waiting time
        int(params["max_time_tour_h"] * 3600),  # maximum time per vehicle [min]
        False,  # Don't force start cumul to zero.
        time,
    )
    time_dimension = routing.GetDimensionOrDie(time)

    # Add time window constraints for each location except depot.
    for location_idx, time_window in enumerate(data_ort["time_windows"]):
        if location_idx == data_ort["depot"]:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])

    # Add time window constraints for each vehicle start node.
    depot_idx = data_ort["depot"]

    for vehicle_id in range(data_ort["num_vehicles"]):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(
            data_ort["time_windows"][depot_idx][0],
            data_ort["time_windows"][depot_idx][1],
        )

        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(vehicle_id))
        )
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.End(vehicle_id))
        )

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print_solution_vrptw(params, data_ort, manager, routing, solution)
    else:
        logger.warning("No solution found!")

    return
