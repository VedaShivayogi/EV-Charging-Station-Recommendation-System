"""
utils.py
--------
Shared geo-utility helpers: haversine distance and vectorized
"distance to nearest existing station" feature engineering.
"""

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance in kilometers between
    (lat1, lon1) and (lat2, lon2). Any argument can be a scalar or
    a numpy array; broadcasting rules apply."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def distance_to_nearest_station(candidates_df, stations_df):
    """For every candidate site, compute the great-circle distance (km)
    to the nearest EXISTING charging station. Returns a numpy array
    aligned with candidates_df's row order."""
    cand_lat = candidates_df["latitude"].to_numpy()[:, None]      # (N, 1)
    cand_lon = candidates_df["longitude"].to_numpy()[:, None]     # (N, 1)
    st_lat = stations_df["latitude"].to_numpy()[None, :]          # (1, M)
    st_lon = stations_df["longitude"].to_numpy()[None, :]         # (1, M)

    dist_matrix = haversine_km(cand_lat, cand_lon, st_lat, st_lon)  # (N, M)
    return dist_matrix.min(axis=1)


def distance_to_point_km(lat_arr, lon_arr, point_lat, point_lon):
    return haversine_km(lat_arr, lon_arr, point_lat, point_lon)
