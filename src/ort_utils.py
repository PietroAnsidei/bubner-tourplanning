"""OrTools utilities."""
from datetime import date, datetime, timedelta

from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def create_data_model(
    num_vehicles, distance_matrix=None, durations_matrix=None, time_windows=None
):
    """Prepare data for OrTools."""
    data = {
        "distance_matrix": distance_matrix,
        "time_matrix": durations_matrix,
        "time_windows": time_windows,
        "num_vehicles": num_vehicles,
        "depot": 0,
    }
    return data


def print_solution_distances(no_to_customer_id_dict, data, manager, routing, solution):
    """Print solution on console."""
    print(f"Objective: {solution.ObjectiveValue()}")
    max_route_distance = 0
    sum_distances = 0

    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0

        while not routing.IsEnd(index):
            plan_output += f" {manager.IndexToNode(index)} ({no_to_customer_id_dict[manager.IndexToNode(index)]}) -> "
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )

        plan_output += f"{manager.IndexToNode(index)} ({no_to_customer_id_dict[manager.IndexToNode(index)]})\n"
        plan_output += f"Distance of the route: {route_distance / 1000} km\n"
        print(plan_output)
        max_route_distance = max(route_distance, max_route_distance)
        sum_distances += route_distance

    print(f"Maximum of the route distances: {max_route_distance / 1000} km")
    print(f"Sum of the route distances: {sum_distances/1000} km")

    return


def solve_distances(params, distances):
    """Solve distance problem."""
    # Instantiate the data problem.
    data_ort = create_data_model(params["num_vehicles"], distance_matrix=distances)

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

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = "Distance"
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        params["max_distance_vehicles_km"]
        * 1000,  # vehicle maximum travel distance [m]
        True,  # start loss from zero
        dimension_name,
    )
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(params["cost_coefficient"])

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print_solution_distances(
            params["stop_id_map"], data_ort, manager, routing, solution
        )
    else:
        print("No solution found !")

    return


def print_solution_durations(params, data, manager, routing, solution):
    """Print solution on console."""
    print(f"Objective: {solution.ObjectiveValue()}")
    time_dimension = routing.GetDimensionOrDie("Time")
    total_time = 0

    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"

        while not routing.IsEnd(index):
            time_var = time_dimension.CumulVar(index)
            time_min = datetime.combine(date.min, params["leave_time"]) + timedelta(
                seconds=solution.Min(time_var)
            )
            time_max = datetime.combine(date.min, params["leave_time"]) + timedelta(
                seconds=solution.Max(time_var)
            )

            plan_output += (
                f" {params['stop_id_map'][manager.IndexToNode(index)]} "
                f"Time({time_min.time()}, {time_max.time()})) -> "
            )
            index = solution.Value(routing.NextVar(index))

        time_var = time_dimension.CumulVar(index)
        time_min = datetime.combine(date.min, params["leave_time"]) + timedelta(
            seconds=solution.Min(time_var)
        )
        time_max = datetime.combine(date.min, params["leave_time"]) + timedelta(
            seconds=solution.Max(time_var)
        )

        plan_output += (
            f" {params['stop_id_map'][manager.IndexToNode(index)]} "
            f"Time({time_min.time()}, {time_max.time()}))\n"
        )
        plan_output += f"Time of the route: {int(solution.Min(time_var) / 60)} min\n"
        print(plan_output)

        total_time += solution.Min(time_var)

    print(f"Sum of the route durations: {int(total_time / 60)} min")

    return


def solve_durations(params, durations):
    """Solve distance problem."""
    # Instantiate the data problem.
    data_ort = create_data_model(
        params["num_vehicles"],
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
        print_solution_durations(params, data_ort, manager, routing, solution)
    else:
        print("No solution found !")

    return
