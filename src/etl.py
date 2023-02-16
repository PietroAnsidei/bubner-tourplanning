"""Logic."""
import os
import pickle
from datetime import date, datetime, timedelta
from random import randrange

import folium
import numpy as np
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
        msg = f"Address {address} not found."
        address = ", ".join(address.split(", ")[1:])
        msg += f" Replacing with {address}."
        logger.warning(msg)
        location = geolocator.geocode(address)

    return location.latitude, location.longitude


def data_import(params, day_id):
    """Import and preprocess data."""
    logger.info(f"Importing {day_id} data from source.")

    # Import data
    stops_df = pd.read_excel(
        f"{params['data_folder']}/{params['data_file']}",
        sheet_name=day_id,
        engine="openpyxl",
        converters={"Customer ID": str, "Postal Code": str, "Dependency": int},
    )

    # Filter data by options
    df = stops_df.copy()

    # Filter customer_id
    if len(params["id_to_remove"]):
        df = df[~df["Customer ID"].isin(params["id_to_remove"])].copy()

    if len(params["pickup_to_remove"]):
        df = df[
            ~(
                (df.Type == "Pickup")
                & (df["Customer ID"].isin(params["pickup_to_remove"]))
            )
        ].copy()

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
            + timedelta(hours=params["pickup_opening_delay_h"])
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
    df.reset_index(inplace=True)
    df.to_pickle(f"{params['data_folder']}/{params['data_locations']}_{day_id}.pkl")
    with open(
        f"{params['data_folder']}/{params['data_distances']}_{day_id}.pkl", "wb"
    ) as handle:
        pickle.dump(response, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return df, response


def data_load(params, day_id):
    """Load data from storage."""
    logger.info(f"Loading {day_id} data from storage.")
    # Fetch data
    df = pd.read_pickle(
        f"{params['data_folder']}/{params['data_locations']}_{day_id}.pkl"
    )

    # Fetch response
    with open(
        f"{params['data_folder']}/{params['data_distances']}_{day_id}.pkl", "rb"
    ) as handle:
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


def data_etl(params, day_id):
    """Preprocess data."""
    # Import data from file (if existing and not required differently)
    if (
        os.path.exists(
            f"{params['data_folder']}/{params['data_locations']}_{day_id}.pkl"
        )
        and os.path.exists(
            f"{params['data_folder']}/{params['data_distances']}_{day_id}.pkl"
        )
        and not params["reload"]
    ):
        df, response = data_load(params, day_id)
    else:
        df, response = data_import(params, day_id)

    # Define leave time
    params["leave_time"] = (
        params["leave_times"][1] if "2" in day_id else params["leave_times"][0]
    )

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

    return params, df, distances, durations


def output_solution(params, day_id, stop_df, routing):
    """Output routing solution to file."""
    if "solution" in routing and len(routing["solution"]):
        # Define output file
        writer = pd.ExcelWriter(f"{params['data_folder']}/{day_id}.xlsx")

        # Create map
        depot_coordinates = [
            float(n_string) for n_string in params["start_loc"][0].split(",")
        ][::-1]
        routing_map = folium.Map(location=depot_coordinates)

        # Add depot icon to map
        folium.Marker(
            location=depot_coordinates,
            tooltip="Depot",
            icon=folium.Icon(icon="fa-solid fa-warehouse", prefix="fa"),
        ).add_to(routing_map)
        visited_IDs = ["depot"]

        # Export each tour separately
        for idx_tour, tour_df in enumerate(routing["solution"]):

            # Create one sheet per tour
            tour_df.to_excel(writer, sheet_name=f"Tour {idx_tour+1}")

            # Initialize tour paths
            last_coordinates = depot_coordinates
            last_ETD = datetime.combine(date.today(), params["leave_time"])
            last_leg_distance = 0

            # Draw tour path map with a random color
            tour_color = "#" + hex(randrange(0, 2**24))[2:]
            for idx_stop, stop in tour_df.iterrows():

                # Identify stop ID, location, ETA and last leg duration
                curr_stop_ID = stop["Stop ID"]
                curr_coordinates = (
                    depot_coordinates
                    if curr_stop_ID == "depot"
                    else stop_df[stop_df["Customer ID"] == curr_stop_ID][
                        ["Latitude", "Longitude"]
                    ]
                    .iloc[0, :]
                    .tolist()
                )
                curr_ETA = (
                    last_ETD
                    if idx_stop == 0
                    else datetime.combine(date.today(), stop.ETA)
                )
                last_duration = int(
                    np.round((curr_ETA - last_ETD).total_seconds() / 60)
                )

                # Add customer icon to map if not already visited
                if curr_stop_ID not in visited_IDs:
                    folium.Marker(
                        location=curr_coordinates,
                        tooltip=f"Customer # {curr_stop_ID}",
                        icon=folium.Icon(icon="fa-solid fa-shop", prefix="fa"),
                    ).add_to(routing_map)

                # Add path between last and current leg
                if curr_coordinates != last_coordinates:
                    msg = f"Tour # {idx_tour+1} Leg # {idx_stop} - {last_leg_distance:.2f} km - {last_duration} min"
                    folium.PolyLine(
                        locations=[last_coordinates, curr_coordinates],
                        tooltip=msg,
                        color=tour_color,
                    ).add_to(routing_map)

                # Store stop location, ETD and next leg distance
                last_coordinates = curr_coordinates
                last_ETD = (
                    datetime.combine(date.today(), stop.ETD)
                    if stop.ETD == stop.ETD
                    else None
                )
                last_leg_distance = stop.Distance
                visited_IDs.append(curr_stop_ID)

        # Close files
        writer.close()
        routing_map.save(f"{params['data_folder']}/{day_id}.html")

    return
