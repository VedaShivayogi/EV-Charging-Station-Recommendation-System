"""
recommender.py
---------------
Turns a per-site suitability score into a final, spatially-sensible
shortlist of N new charging-station locations.

Why not just take the top-N scores directly?
Because suitability surfaces are spatially smooth (a great location
is usually surrounded by other near-great locations), naive top-N
selection tends to cluster every recommendation into a single
neighborhood -- useless for a city that needs coverage. We fix this
with a greedy max-coverage-style heuristic (a simplified, fast
approximation of the classic p-median / facility-location problem):

    1. Discard candidates that are already too close to an existing
       station (`min_dist_existing_km`) -- no point recommending a
       site 200m from a station that already exists.
    2. Sort remaining candidates by predicted suitability score,
       descending.
    3. Walk down the sorted list, greedily accepting a candidate only
       if it is at least `min_dist_between_new_km` away from every
       already-accepted recommendation. This enforces spatial spread
       while still favoring higher-scored sites first.
    4. Stop once N sites are selected (or candidates run out).

This is a well-known, fast (O(N*k)) greedy approximation used in
facility-location / sensor-placement literature; it won't find the
mathematically optimal set (that's NP-hard in general), but it gives
good, explainable, real-time results -- appropriate for an
interactive planning tool.
"""

import numpy as np
import pandas as pd

from src.utils import haversine_km


def filter_oversaturated(df, min_dist_existing_km):
    return df[df["dist_to_nearest_station_km"] >= min_dist_existing_km].copy()


def greedy_spatial_select(df, n_sites, min_dist_between_new_km, score_col="predicted_score"):
    """
    df must contain latitude, longitude, and score_col.
    Returns a DataFrame of the selected rows, in selection order, with
    an added `rank` column.
    """
    candidates = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    selected_idx = []
    selected_lat = []
    selected_lon = []

    for idx, row in candidates.iterrows():
        if len(selected_idx) >= n_sites:
            break
        if selected_lat:
            dists = haversine_km(
                np.array(selected_lat), np.array(selected_lon),
                row["latitude"], row["longitude"],
            )
            if dists.min() < min_dist_between_new_km:
                continue
        selected_idx.append(idx)
        selected_lat.append(row["latitude"])
        selected_lon.append(row["longitude"])

    result = candidates.loc[selected_idx].copy()
    result.insert(0, "rank", range(1, len(result) + 1))
    return result


def recommend_new_stations(scored_df, n_sites=10, min_dist_existing_km=1.2,
                            min_dist_between_new_km=1.5):
    """Full pipeline: filter oversaturated areas, then greedily select
    spatially-diverse, high-scoring sites."""
    filtered = filter_oversaturated(scored_df, min_dist_existing_km)
    return greedy_spatial_select(filtered, n_sites, min_dist_between_new_km)
