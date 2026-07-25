"""
app.py
------
Streamlit dashboard for the EV Charging Station Recommendation System.
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_folium import st_folium

from src.pipeline import run_full_pipeline
from src.mapping import build_map
from src.data_processing import DEFAULT_WEIGHTS

st.set_page_config(page_title="EV Charging Station Recommender", layout="wide", page_icon="🔌")

st.title("🔌 EV Charging Station Recommendation System")
st.caption(
    "Analyzes population density, traffic, points-of-interest and existing charger "
    "coverage to recommend where new EV charging stations should be built."
)

# ------------------------- Sidebar controls -------------------------
with st.sidebar:
    st.header("Configuration")
    city = st.selectbox("City", ["Bengaluru", "Delhi", "Mumbai"], index=0)
    grid_size = st.slider("Candidate grid resolution", 20, 60, 40, step=5,
                           help="Higher = more candidate sites analyzed, slower to run.")
    n_existing_stations = st.slider("Existing stations (simulated)", 10, 60, 25)
    n_new_sites = st.slider("Number of new stations to recommend", 3, 25, 10)

    st.subheader("Spatial constraints")
    min_dist_existing_km = st.slider("Min. distance from an existing station (km)", 0.2, 5.0, 1.2, 0.1)
    min_dist_between_new_km = st.slider("Min. distance between new recommendations (km)", 0.5, 5.0, 1.5, 0.1)

    st.subheader("Scoring weights")
    st.caption("Controls how the *training label* balances demand vs. accessibility. "
               "The Random Forest still has to learn the pattern from raw features.")
    w_population = st.slider("Population density weight", 0.0, 1.0, DEFAULT_WEIGHTS["population"])
    w_traffic = st.slider("Traffic volume weight", 0.0, 1.0, DEFAULT_WEIGHTS["traffic"])
    w_poi = st.slider("Commercial / POI density weight", 0.0, 1.0, DEFAULT_WEIGHTS["poi"])
    w_income = st.slider("Income index weight", 0.0, 1.0, DEFAULT_WEIGHTS["income"])
    w_competition = st.slider("Under-served bonus weight", 0.0, 1.0, DEFAULT_WEIGHTS["competition"])
    w_center = st.slider("Distance-from-center penalty", 0.0, 1.0, DEFAULT_WEIGHTS["center_penalty"])

    n_clusters = st.slider("Demand clusters (KMeans, k)", 2, 8, 5)

    run_btn = st.button("🚀 Generate Recommendations", type="primary", use_container_width=True)

if "result" not in st.session_state:
    st.session_state.result = None

if run_btn or st.session_state.result is None:
    weights = {
        "population": w_population, "traffic": w_traffic, "poi": w_poi,
        "income": w_income, "competition": w_competition, "center_penalty": w_center,
    }
    with st.spinner("Analyzing city data and training model..."):
        st.session_state.result = run_full_pipeline(
            city=city, grid_size=grid_size, n_existing_stations=n_existing_stations,
            n_new_sites=n_new_sites, min_dist_existing_km=min_dist_existing_km,
            min_dist_between_new_km=min_dist_between_new_km, weights=weights,
            n_clusters=n_clusters,
        )
    st.session_state.min_dist_between_new_km = min_dist_between_new_km

result = st.session_state.result

# ------------------------- Top metrics -------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Candidate sites analyzed", len(result["candidates"]))
col2.metric("Existing stations", len(result["stations"]))
col3.metric("New sites recommended", len(result["recommendations"]))
col4.metric("Model R² (held-out test)", result["model_metrics"]["r2"])

# ------------------------- Map -------------------------
st.subheader("📍 Interactive Map")
st.caption("Red = existing stations · Green bolt = recommended new station · "
           "Heat layer = combined demand (population + traffic + POI)")
fmap = build_map(
    result["candidates"], result["stations"], result["recommendations"],
    min_dist_between_new_km=st.session_state.get("min_dist_between_new_km", 1.5),
)
st_folium(fmap, width=None, height=560, returned_objects=[])

# ------------------------- Tabs -------------------------
tab1, tab2, tab3 = st.tabs(["🏆 Recommendations", "📊 Model Insights", "🧭 Demand Clusters"])

with tab1:
    st.subheader("Recommended new station sites")
    display_cols = ["rank", "site_id", "latitude", "longitude", "predicted_score",
                     "population_density", "traffic_volume", "poi_score",
                     "dist_to_nearest_station_km", "demand_cluster"]
    st.dataframe(
        result["recommendations"][display_cols].style.format({
            "latitude": "{:.4f}", "longitude": "{:.4f}", "predicted_score": "{:.1f}",
            "population_density": "{:.0f}", "traffic_volume": "{:.0f}",
            "poi_score": "{:.1f}", "dist_to_nearest_station_km": "{:.2f}",
        }),
        use_container_width=True, hide_index=True,
    )
    csv = result["recommendations"][display_cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download recommendations as CSV", csv,
                        "recommended_stations.csv", "text/csv")

with tab2:
    st.subheader("Random Forest performance")
    m1, m2, m3 = st.columns(3)
    m1.metric("R²", result["model_metrics"]["r2"])
    m2.metric("MAE", result["model_metrics"]["mae"])
    m3.metric("Train / Test rows", f"{result['model_metrics']['n_train']} / {result['model_metrics']['n_test']}")

    st.subheader("Feature importance")
    fi = result["feature_importances"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(fi["feature"][::-1], fi["importance"][::-1], color="#2ecc71")
    ax.set_xlabel("Importance")
    ax.set_title("What drives the suitability score?")
    fig.tight_layout()
    st.pyplot(fig)

with tab3:
    st.subheader("KMeans demand-cluster profiles")
    st.caption("Average feature values per unsupervised demand cluster — helps label "
               "each cluster (e.g. 'dense downtown', 'suburban', 'highway commercial').")
    st.dataframe(result["cluster_profile"], use_container_width=True)

    cluster_counts = result["candidates"]["demand_cluster"].value_counts().sort_index()
    fig2, ax2 = plt.subplots(figsize=(7, 3))
    ax2.bar(cluster_counts.index.astype(str), cluster_counts.values, color="#3498db")
    ax2.set_xlabel("Cluster")
    ax2.set_ylabel("Number of candidate sites")
    ax2.set_title("Candidate sites per demand cluster")
    fig2.tight_layout()
    st.pyplot(fig2)

st.divider()
st.caption("EV Charging Station Recommendation System · Synthetic demo data · "
           "Built with Python, Pandas, Scikit-learn, Folium & Streamlit.")
