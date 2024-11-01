"""Main optimizer."""
from src.etl import data_etl, output_solution
from src.optimization import repeat_tests, solve_vr
from src.setup import setup_params

if __name__ == "__main__":
    # Import parameters
    params = setup_params()

    # Process by day_ID
    for day_id in params["data_sheet"]:
        # ETL
        params, df, distances, durations = data_etl(params, day_id)

        # Solver
        if params["test_mode"]:
            routing = repeat_tests(params, distances, durations)
        else:
            routing = solve_vr(params, distances, durations)

        # Output results to file
        output_solution(params, day_id, df, routing)
