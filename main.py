"""Main optimizer."""
from src.logic import data_etl
from src.ort_utils import solve_vr
from src.setup import setup_params

if __name__ == "__main__":
    # Import parameters
    params = setup_params()

    # ETL
    params, distances, durations = data_etl(params)

    # Solver
    solve_vr(params, distances, durations)

# TODO: cover the following use cases:
#  - add pickup constraints (should happen after delivery) + max 4 legs (?)
#  - add customer delivery
