"""
data_processing.py
-------------------
Feature engineering: turns the raw candidate-site table + existing
station table into a model-ready feature matrix, and (for training
purposes only) a synthetic "ground-truth" suitability label.

In a production setting you would replace `build_training_labels`
with real outcome data, e.g. observed utilization of stations already
built, or expert-scored site audits. Here we simulate that signal
with a transparent weighted formula plus noise, purely so the
regression model has something realistic to learn from -- the model
still has to recover the (noisy, nonlinear-ized) relationship from
the raw features, it isn't just copying the formula.
"""

import numpy as np
import pandas as pd

from src.utils import distance_to_nearest_station, distance_to_point_km

FEATURE_COLUMNS = [
    "population_density",
    "traffic_volume",
    "poi_score",
    "income_index",
    "dist_to_nearest_station_km",
    "dist_to_city_center_km",
]


def engineer_features(candidates_df, stations_df):
    """Adds engineered columns to a copy of candidates_df and returns it."""
    df = candidates_df.copy()

    df["dist_to_nearest_station_km"] = distance_to_nearest_station(df, stations_df)

    city_center_lat = candidates_df["latitude"].mean()
    city_center_lon = candidates_df["longitude"].mean()
    df["dist_to_city_center_km"] = distance_to_point_km(
        df["latitude"].to_numpy(), df["longitude"].to_numpy(),
        city_center_lat, city_center_lon,
    )
    return df


def build_training_labels(df, weights, seed=7):
    """
    Builds a synthetic 0-100 'suitability score' used ONLY to train the
    demonstration model. Combines normalized demand-side features
    (population, traffic, POI, income) positively, and supply-side /
    accessibility features (distance to an existing charger, distance
    to center) with a diminishing-returns transform -- being far from
    an existing charger helps up to a point, then stops mattering
    (nobody needs a charger 40km from anything), and being far from
    the city center is mildly penalized (harder to build/operate).
    """
    rng = np.random.default_rng(seed)

    def norm(col):
        x = df[col].to_numpy(dtype=float)
        return (x - x.min()) / (x.max() - x.min() + 1e-9)

    demand = (
        weights["population"] * norm("population_density")
        + weights["traffic"] * norm("traffic_volume")
        + weights["poi"] * norm("poi_score")
        + weights["income"] * norm("income_index")
    )

    # Diminishing-returns "under-served" bonus: sqrt caps the benefit
    # of being extremely far from any existing station.
    competition_gap = np.sqrt(np.clip(df["dist_to_nearest_station_km"], 0, None))
    competition_gap = competition_gap / (competition_gap.max() + 1e-9)

    center_penalty = norm("dist_to_city_center_km")

    raw_score = (
        demand
        + weights["competition"] * competition_gap
        - weights["center_penalty"] * center_penalty
    )
    raw_score = raw_score + rng.normal(0, 0.05, size=len(df))  # label noise
    raw_score = (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min() + 1e-9)
    return raw_score * 100


DEFAULT_WEIGHTS = {
    "population": 0.30,
    "traffic": 0.25,
    "poi": 0.20,
    "income": 0.10,
    "competition": 0.20,
    "center_penalty": 0.08,
}
