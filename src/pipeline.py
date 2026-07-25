"""
pipeline.py
-----------
Single entry point used by both app.py (Streamlit) and run_pipeline.py
(CLI) so the two never drift out of sync.
"""

from data.generate_data import generate_city_data
from src.data_processing import engineer_features, build_training_labels, DEFAULT_WEIGHTS
from src.model import SuitabilityModel, DemandClustering
from src.recommender import recommend_new_stations


def run_full_pipeline(city="Bengaluru", grid_size=45, n_existing_stations=25,
                       n_new_sites=10, min_dist_existing_km=1.2,
                       min_dist_between_new_km=1.5, weights=None,
                       n_clusters=5, seed=42):
    weights = weights or DEFAULT_WEIGHTS

    candidates_raw, stations_df = generate_city_data(
        city=city, grid_size=grid_size, n_existing_stations=n_existing_stations, seed=seed
    )
    features_df = engineer_features(candidates_raw, stations_df)

    # Synthetic label for demo training (see data_processing.py docstring)
    label = build_training_labels(features_df, weights)

    model = SuitabilityModel().fit(features_df, label)
    features_df["predicted_score"] = model.predict(features_df)

    clustering = DemandClustering(n_clusters=n_clusters)
    features_df["demand_cluster"] = clustering.fit_predict(features_df)
    cluster_profile = clustering.cluster_profile(features_df, features_df["demand_cluster"])

    recommendations = recommend_new_stations(
        features_df,
        n_sites=n_new_sites,
        min_dist_existing_km=min_dist_existing_km,
        min_dist_between_new_km=min_dist_between_new_km,
    )

    return {
        "candidates": features_df,
        "stations": stations_df,
        "recommendations": recommendations,
        "model": model,
        "model_metrics": model.metrics_,
        "feature_importances": model.feature_importances(),
        "cluster_profile": cluster_profile,
    }
