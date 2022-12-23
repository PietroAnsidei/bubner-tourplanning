"""Main optimizer."""
from src.logic import data_etl
from src.ort_utils import solve_vrp
from src.setup import setup_params

if __name__ == "__main__":
    # Import parameters
    params = setup_params()

    # ETL
    params, distances = data_etl(params)

    # Solver
    solve_vrp(params, distances)
