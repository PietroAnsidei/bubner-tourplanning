"""Main optimizer."""
from src.etl import data_etl, output_solution
from src.optimization import repeat_tests, solve_vr
from src.setup import setup_params

if __name__ == "__main__":
    # Import parameters
    params = setup_params()

    # ETL
    params, distances, durations = data_etl(params)

    # Solver
    if params["test_mode"]:
        routing = repeat_tests(params, distances, durations)
    else:
        routing = solve_vr(params, distances, durations)

    # Output results to file
    output_solution(params, routing)
