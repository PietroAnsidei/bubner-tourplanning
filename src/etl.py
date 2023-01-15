"""Logic."""
import os
import pickle
from datetime import date, datetime, timedelta

import pandas as pd
import requests
from geopy.geocoders import Nominatim

from src.setup import logger


def address_to_lat_lon(address):
    """Fetch latitude and longitude string from address."""
    # Import nominatim for address research
    geolocator = Nominatim(user_agent="Bubner_Tourplanning")

    # Execute query
    location = geolocator.geocode(address)

    # In case of failure with the address, just add the town
    if location is None:
        address = ", ".join(address.split(", ")[1:])
        location = geolocator.geocode(address)

    return location.latitude, location.longitude


def data_import(params):
    """Import and preprocess data."""
    logger.info("Importing data from source.")

    # Import data
    stops_df = pd.read_excel(
        f"{params['data_folder']}/{params['data_file']}",
        sheet_name=params["data_sheet"],
        engine="openpyxl",
        converters={"Customer ID": str, "Postal Code": str, "Dependency": int},
    )

    # Filter data by options
    df = stops_df.copy()

    # Extract only delivery
    if params["only_delivery"]:
        df = df[df.Type == "Delivery"].copy()

    # Filter only stores and not customer delivery
    if params["only_stores"]:
        df = df[df["Stop type"] == "Store"].copy()

    # Add service time along the route
    service_duration_s = (
        df["Duration (in min)"]
        .apply(
            lambda t: int(
                timedelta(
                    hours=t.hour, minutes=t.minute, seconds=t.second
                ).total_seconds()
            )
        )
        .rename("service_duration_s")
    )
    df = df.join(service_duration_s).drop("Duration (in min)", axis=1)

    # Add capacity requests
    df["capacity"] = df["Stop type"].replace({"Customer Delivery": -0.1, "Store": -1})
    df.loc[df.Type == "Pickup", "capacity"] = 0.01

    # Add pickup time requests
    df["delayed_time_to"] = df["Time to"].apply(
        lambda x: (
            datetime.combine(datetime.today(), x)
            + timedelta(hours=params["pickup_delay_h"])
        ).time()
    )
    df.loc[df.Type == "Pickup", "Time to"] = df.loc[
        df.Type == "Pickup", "delayed_time_to"
    ]

    # Extract Latitude and Longitude data
    # Customers: fetch Latitude and Longitude data from full address
    df["Municipality"] = df["Town"].apply(lambda x: x.split("OT")[0])
    df["full_address"] = df[["Address", "Postal Code", "Municipality"]].agg(
        ", ".join, axis=1
    )
    lat_lon_addr = pd.DataFrame(
        df.full_address.apply(lambda x: address_to_lat_lon(x)).tolist(),
        index=df.index,
        columns=["Latitude", "Longitude"],
    )

    # Stores: split Latitude and Longitude and store them in df
    lat_lon_str = pd.DataFrame(
        df["Latitude, Longitude"]
        .apply(lambda x: x.split(", ") if isinstance(x, str) else (x, x))
        .tolist(),
        index=df.index,
        columns=["Latitude", "Longitude"],
    ).astype(float)

    lat_lon = lat_lon_str.fillna(lat_lon_addr)

    # Merge dataframes
    df = df.join(lat_lon).drop("Latitude, Longitude", axis=1)

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
    logger.info("Loading data from storage.")
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
    # Import data from file (if existing and not required differently)
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

    params["service_time"] = [0] + df.service_duration_s.tolist()

    # Capacity
    params["demands"] = [params["max_legs"]] + df.capacity.tolist()

    # Pickup dependency (notice that delivery comes first). Add 1 for the depot index.
    params["pickups"] = [
        [df.index[df.No == loc.Dependency][0] + 1, idx + 1]
        for idx, loc in df.iterrows()
        if loc.Type == "Pickup"
    ]

    return params, distances, durations
