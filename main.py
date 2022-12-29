"""Main optimizer."""
from src.logic import data_etl
from src.ort_utils import solve_vrp, solve_vrptw
from src.setup import setup_params

if __name__ == "__main__":
    # Import parameters
    params = setup_params()

    # ETL
    params, distances, durations = data_etl(params)

    # Solver
    if params["solve_by_distance"]:
        solve_vrp(params, distances)
    else:
        solve_vrptw(params, durations)
