"""Main optimizer."""
from src.logic import data_etl
from src.ort_utils import solve_distances, solve_durations
from src.setup import setup_params

if __name__ == "__main__":
    # Import parameters
    params = setup_params()

    # ETL
    params, distances, durations = data_etl(params)

    # Solver
    if params["solve_by_distance"]:
        solve_distances(params, distances)
    else:
        solve_durations(params, durations)
