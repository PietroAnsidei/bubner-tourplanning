"""Constants."""

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
}
