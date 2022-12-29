"""Logic."""
import os
import pickle
from datetime import date, datetime

import pandas as pd
import requests


def data_import(params):
    """Import and preprocess data."""
    # Import data
    stops_df = pd.read_excel(
        f"{params['data_folder']}/{params['data_file']}",
        sheet_name=params["data_sheet"],
        engine="openpyxl",
    )

    # Extract only store-delivery and check uniqueness
    df = stops_df[(stops_df.Type == "Delivery") & (stops_df["Stop type"] == "Store")]

    unique_customers = df["Customer ID"].unique()
    assert len(unique_customers) == df.shape[0]

    # Split Latitude and Longitude and store them in df
    lat_lon = pd.DataFrame(
        df["Latitude, Longitude"].apply(lambda x: x.split(", ")).tolist(),
        index=df.index,
        columns=["Latitude", "Longitude"],
    ).astype(float)
    df = df.join(lat_lon).drop("Latitude, Longitude", axis=1).reset_index()

    # The format for OSRM is longitude/latitude, not latitude/longitude
    req_str = ";".join(
        params["start_loc"]
        + (df.Longitude.astype(str) + "," + df.Latitude.astype(str)).tolist()
    )

    # https://project-osrm.org/docs/v5.22.0/api/#general-options
    response = requests.get(
        f"{params['url_distance']}{req_str}", params=params["osrm_params"]
    )

    # Store data & response
    df.to_pickle(f"{params['data_folder']}/{params['data_locations']}")
    with open(f"{params['data_folder']}/{params['data_distances']}", "wb") as handle:
        pickle.dump(response, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return df, response


def data_load(params):
    """Load data from storage."""
    # Fetch data
    df = pd.read_pickle(f"{params['data_folder']}/{params['data_locations']}")

    # Fetch response
    with open(f"{params['data_folder']}/{params['data_distances']}", "rb") as handle:
        response = pickle.load(handle)

    return df, response


def delay_from_leave_time(params, ts):
    """Compare timestamp with leave time and return difference in minutes."""
    return int(
        (
            datetime.combine(date.min, max(ts, params["leave_time"]))
            - datetime.combine(date.min, params["leave_time"])
        ).seconds
    )


def data_etl(params):
    """Preprocess data."""
    if (
        os.path.exists(f"{params['data_folder']}/{params['data_locations']}")
        and os.path.exists(f"{params['data_folder']}/{params['data_distances']}")
        and not params["reload"]
    ):
        df, response = data_load(params)
    else:
        df, response = data_import(params)

    # Extract store info-id
    params["stop_id_map"] = {
        **{0: "depot"},
        **pd.Series(df["Customer ID"].values, index=1 + df.index).to_dict(),
    }

    # Distance and time matrices - convert to int ([m] and [s])
    distances = [[int(j) for j in i] for i in response.json()["distances"]]
    durations = [[int(j) for j in i] for i in response.json()["durations"]]

    # Time windows
    df["delay_reach"] = (
        df["Time from"].apply(lambda x: delay_from_leave_time(params, x)).tolist()
    )
    df["delay_leave"] = (
        df["Time to"].apply(lambda x: delay_from_leave_time(params, x)).tolist()
    )

    params["time_windows"] = [
        (0, int(params["max_time_tour_h"] * 3600)),  # depot
    ] + list(df[["delay_reach", "delay_leave"]].itertuples(index=False, name=None))

    return params, distances, durations
