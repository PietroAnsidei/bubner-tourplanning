"""Constants."""
from datetime import time

params = {
    # Define paths
    "data_folder": "data",
    "data_file": "Data.xlsx",
    "data_locations": "osrm_data.pkl",
    "data_distances": "osrm_response.pkl",
    # ETL
    "reload": True,
    "data_sheet": "Stops 1.Tour (Mo-Fr)",
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
    "leave_time": time(2, 45),
    "max_time_tour_h": 7,
    "pickup_delay_h": 0,  # Allow max delay on the ETA (Time to) for pickup
    # Other parameters
    "max_distance_vehicles_km": 1000,  # For each vehicle's route
    "pen_distance_vehicle_factor": 0,  # To penalize the max of the routes lengths
    "slack_time_max_h": 0.5,  # Max allowed stopping time at location + min pickup delay
}
