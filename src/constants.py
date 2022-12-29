"""Constants."""
from datetime import time

params = {
    # Define paths
    "data_folder": "data",
    "data_file": "Data.xlsx",
    "data_locations": "osrm_data.pkl",
    "data_distances": "osrm_reponse.pkl",
    # ETL
    "data_sheet": "Stops 1.Tour (Mo-Fr)",
    # Coordinates of the production centre at Gewerbegebiet Südstraße 5
    "start_loc": ["13.5853717,51.6286970"],
    "osrm_params": {"annotations": "distance,duration"},
    # URLs
    "url_distance": "http://router.project-osrm.org/table/v1/driving/",
    # Options
    "reload": True,
    "solve_by_distance": True,
    "num_vehicles": 7,
    "leave_time": time(2, 45),
    "max_distance_vehicles_km": 1000,
    "cost_coefficient": 100,
    "max_time_tour_h": 2.75,
    "pen_distance_vehicle_factor": 0,
    "slack_time_min": 30,
    "max_legs": 4,
}
