"""Logic."""
import os
import pickle

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

    # Distance and time matrices
    distances = response.json()["distances"]
    # durations = response.json()["durations"]

    return params, distances
