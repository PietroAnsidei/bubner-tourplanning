"""Constants."""
from datetime import time

params = {
    # Test mode
    "test_mode": False,
    "num_test": 1000,
    "verbose": True,
    "strategies": [
        # 'AUTOMATIC',  # works
        "PATH_CHEAPEST_ARC",  # timeout
        "PATH_MOST_CONSTRAINED_ARC",  # works
        "PARALLEL_CHEAPEST_INSERTION",  # works
        "LOCAL_CHEAPEST_INSERTION",  # works
        "GLOBAL_CHEAPEST_ARC",  # timeout
        "LOCAL_CHEAPEST_ARC",  # timeout
        "FIRST_UNBOUND_MIN_VALUE",  # timeout
    ],
    "local_search": [
        # 'AUTOMATIC',  # works
        "GREEDY_DESCENT",  # works
        "GUIDED_LOCAL_SEARCH",  # timeout
        "SIMULATED_ANNEALING",  # timeout
        "TABU_SEARCH",  # timeout
        "GENERIC_TABU_SEARCH",  # timeout
    ],
    # Define paths
    "data_folder": "data",
    "data_file": "Data collection_v4.xlsx",
    "data_locations": "osrm_data",
    "data_distances": "osrm_response",
    "output_test_file": "Test.csv",
    "output_test_plot": "Test_plot.png",
    # ETL
    "reload": True,
    "data_sheet": [
        "Stops 1.Tour (Mo-Fr)",
        "Stops 2.Tour (Mo-Fr)",
        "Saturday 1. Tour",
        "Saturday 2.Tour",
        "Sunday tours",
    ],
    "only_stores": False,
    "only_delivery": False,
    # Coordinates of the production centre at Gewerbegebiet Südstraße 5
    "start_loc": ["13.5853717,51.6286970"],
    "osrm_params": {"annotations": "distance,duration"},
    # URLs
    "url_distance": "http://router.project-osrm.org/table/v1/driving/",
    # Optimization parameters
    # Solve routing either minimizing travel distance or duration
    "solve_by_distance": True,
    # Add constraint for visiting locations within time windows and/or with capacity
    "constrain_tw": True,
    "constrain_cap": True,
    # Maximum execution time
    "max_search_time_min": 10,
    # Requirements
    "num_vehicles": 10,
    "max_legs": 4.44,  # Allow 4 store- (1), 4 customer- (0.1), 4 pickup-legs(0.01)
    "leave_times": [time(2, 45), time(8, 00)],
    "max_time_tour_h": 9,
    "pickup_opening_delay_h": 6,  # Allow max delay on the ETA (Time to) for pickup
    # Other parameters
    "max_distance_vehicles_km": 2000,  # For each vehicle's route
    "pen_distance_vehicle_factor": 0,  # To penalize the max of the routes lengths
    "slack_time_max_h": 4,  # Max allowed stopping time at location
    "min_pickup_delay_h": 0.5,  # min pickup delay
    # Comparison health check
    "id_to_remove": [],  # ["141", "142", "143", "144", "145", "146"],
    "pickup_to_remove": [],  # ["9023"],
}
