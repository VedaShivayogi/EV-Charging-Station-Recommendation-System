"""
model.py
--------
Two complementary ML components:

1. SuitabilityModel (supervised)
   A RandomForestRegressor trained to predict a 0-100 site-suitability
   score from engineered features. Random Forest was chosen because:
     - it handles nonlinear interactions between features (e.g. "high
       traffic only matters if income is also mid-high") without manual
       feature crosses,
     - it's robust to unscaled / mixed-range features,
     - it gives free, interpretable feature-importance output, which
       matters for a decision-support tool where planners need to
       justify *why* a site was recommended,
     - it trains in milliseconds on a few thousand rows, which keeps
       the Streamlit app interactive.

2. DemandClustering (unsupervised)
   KMeans over standardized demand features to identify a small number
   of distinct "demand regimes" across the city (e.g. dense downtown,
   suburban residential, highway/commercial corridor). This is used as
   a complementary, label-free view: even without any target variable,
   it shows planners where the natural demand hotspots are, and lets
   us report a cluster ID per recommended site for interpretability.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from src.data_processing import FEATURE_COLUMNS


class SuitabilityModel:
    def __init__(self, n_estimators=300, max_depth=10, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        )
        self.metrics_ = {}

    def fit(self, df, target):
        X = df[FEATURE_COLUMNS]
        y = target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        self.metrics_ = {
            "r2": round(r2_score(y_test, preds), 4),
            "mae": round(mean_absolute_error(y_test, preds), 3),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        return self

    def predict(self, df):
        return self.model.predict(df[FEATURE_COLUMNS])

    def feature_importances(self):
        importances = self.model.feature_importances_
        return pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)


class DemandClustering:
    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.cluster_cols = ["population_density", "traffic_volume", "poi_score", "income_index"]

    def fit_predict(self, df):
        X = self.scaler.fit_transform(df[self.cluster_cols])
        labels = self.kmeans.fit_predict(X)
        return labels

    def cluster_profile(self, df, labels):
        profile = df[self.cluster_cols].copy()
        profile["cluster"] = labels
        return profile.groupby("cluster").mean().round(1)
