"""Main optimizer."""
import pickle

import pandas as pd
import requests
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def create_data_model(distances, num_vehicles):
    """Stores the data for the problem."""
    return {"distance_matrix": distances, "num_vehicles": num_vehicles, "depot": 0}


def print_solution(no_to_customer_id_dict, data, manager, routing, solution):
    """Prints solution on console."""
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
    no_to_customer_id_dict,
    distances,
    num_vehicles,
    vehicle_max_dist,  # in km
    distance_global_span_cost_coefficient,
):
    """Entry point of the program."""
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
        """Returns the distance between the two nodes."""
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
        print_solution(no_to_customer_id_dict, data_ort, manager, routing, solution)
    else:
        print("No solution found !")


if __name__ == "__main__":

    # Define paths
    data_folder = "data"
    data_file = "Data.xlsx"
    data_sheet = "Stops 1.Tour (Mo-Fr)"
    url_distance = "http://router.project-osrm.org/table/v1/driving/"
    data_distances = "osrm_reponse.pkl"

    # Import data
    stops_df = pd.read_excel(
        f"{data_folder}/{data_file}", sheet_name=data_sheet, engine="openpyxl"
    )

    # Extract only store-delivery and check uniqueness
    df = stops_df[(stops_df.Type == "Delivery") & (stops_df["Stop type"] == "Store")]

    unique_customers = df["Customer ID"].unique()
    assert len(unique_customers) == df.shape[0]

    # Split Latitude and Longitude and store them in df
    lat_lon = pd.DataFrame(
        df["Latitude, Longitude"].apply(lambda x: x.split(", ")).tolist(),
        index=df.index,
        columns=["Latitude", "Longitude"],
    ).astype(float)
    df = df.join(lat_lon).drop("Latitude, Longitude", axis=1).reset_index()

    # Extract store info-id
    no_to_customer_id_dict = {
        **{0: "depot"},
        **pd.Series(df["Customer ID"].values, index=1 + df.index).to_dict(),
    }

    # First, the coordinates of the production centre at Gewerbegebiet Südstraße 5
    start_loc = ["13.5853717,51.6286970"]

    # The format for OSRM is longitude/latitude, not latitude/longitude
    req_str = ";".join(
        start_loc + (df.Longitude.astype(str) + "," + df.Latitude.astype(str)).tolist()
    )

    # https://project-osrm.org/docs/v5.22.0/api/#general-options
    response = requests.get(
        f"{url_distance}{req_str}", params={"annotations": "distance,duration"}
    )

    # Distance and time matrices
    distances = response.json()["distances"]
    durations = response.json()["durations"]

    # Store distances
    with open(f"{data_folder}/{data_distances}", "wb") as handle:
        pickle.dump(response, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Fetch distances
    with open(f"{data_folder}/{data_distances}", "rb") as handle:
        response = pickle.load(handle)

    solve_distances(no_to_customer_id_dict, distances, 7, 1000, 100)
