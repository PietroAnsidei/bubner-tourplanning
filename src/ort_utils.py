"""OrTools utilities."""
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def create_data_model(distance_matrix, num_vehicles):
    """Prepare data for OrTools."""
    data = {
        "distance_matrix": distance_matrix,
        "num_vehicles": num_vehicles,
        "depot": 0,
    }
    return data


def print_solution(no_to_customer_id_dict, data, manager, routing, solution):
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
        plan_output += "Distance of the route: {}km\n".format(route_distance / 1000)
        print(plan_output)
        max_route_distance = max(route_distance, max_route_distance)
        sum_distances += route_distance
    print("Maximum of the route distances: {}km".format(max_route_distance / 1000))
    print(f"Sum of the route distances: {sum_distances/1000}km")

    return


def solve_distances(
    params,
    distances,
    num_vehicles,
    vehicle_max_dist,  # in km
    distance_global_span_cost_coefficient,
):
    """Solve distance problem."""
    # Instantiate the data problem.
    data_ort = create_data_model(distances, num_vehicles)

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
        vehicle_max_dist * 1000,  # vehicle maximum travel distance
        True,  # start loss from zero
        dimension_name,
    )
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(
        distance_global_span_cost_coefficient
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
        print_solution(params["stop_id_map"], data_ort, manager, routing, solution)
    else:
        print("No solution found !")

    return
