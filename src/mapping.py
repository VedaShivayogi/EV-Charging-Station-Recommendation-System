"""
mapping.py
----------
Builds the interactive Folium map: a demand heatmap layer, existing
station markers (red), and recommended new-station markers (green,
numbered by rank), plus a light-grey circle showing the "exclusion
radius" enforced around each recommendation for transparency.
"""

import folium
from folium.plugins import HeatMap, MarkerCluster


def build_map(candidates_df, stations_df, recommendations_df,
               min_dist_between_new_km=1.5, zoom_start=12):
    center_lat = candidates_df["latitude"].mean()
    center_lon = candidates_df["longitude"].mean()

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start,
                       tiles="CartoDB positron")

    # --- Demand heatmap (population + traffic + poi, normalized) ---
    heat_weight = (
        candidates_df["population_density"] / candidates_df["population_density"].max()
        + candidates_df["traffic_volume"] / candidates_df["traffic_volume"].max()
        + candidates_df["poi_score"] / candidates_df["poi_score"].max()
    ) / 3.0
    heat_data = list(zip(candidates_df["latitude"], candidates_df["longitude"], heat_weight))
    HeatMap(heat_data, radius=14, blur=18, min_opacity=0.25,
            name="Demand heatmap (population + traffic + POI)").add_to(fmap)

    # --- Existing stations ---
    existing_layer = folium.FeatureGroup(name="Existing charging stations")
    for _, row in stations_df.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color="#c0392b",
            fill=True,
            fill_color="#e74c3c",
            fill_opacity=0.9,
            tooltip=(f"Existing station {row['station_id']}<br>"
                     f"Chargers: {row['n_chargers']}<br>"
                     f"Utilization: {row['avg_utilization_pct']}%"),
        ).add_to(existing_layer)
    existing_layer.add_to(fmap)

    # --- Recommended new stations ---
    rec_layer = folium.FeatureGroup(name="Recommended new stations")
    for _, row in recommendations_df.iterrows():
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            icon=folium.Icon(color="green", icon="bolt", prefix="fa"),
            tooltip=(f"Rank #{int(row['rank'])} | Score: {row['predicted_score']:.1f}<br>"
                     f"Pop. density: {row['population_density']:.0f}/km²<br>"
                     f"Traffic: {row['traffic_volume']:.0f} veh/hr<br>"
                     f"Nearest existing station: {row['dist_to_nearest_station_km']:.2f} km"),
        ).add_to(rec_layer)
        folium.Circle(
            location=[row["latitude"], row["longitude"]],
            radius=min_dist_between_new_km * 1000 / 2,
            color="#27ae60",
            weight=1,
            fill=False,
            opacity=0.35,
        ).add_to(rec_layer)
    rec_layer.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap
