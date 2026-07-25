"""
generate_data.py
-----------------
Generates a synthetic but realistic dataset for a city:
  1. A grid of candidate locations covering the city's bounding box.
  2. Population density, traffic volume, and points-of-interest (POI /
     commercial density) surfaces, built from a mixture of Gaussian
     "hotspots" (mimicking real cities where density clusters around
     a downtown core, tech parks, malls, transit hubs, etc.)
  3. A set of existing EV charging stations placed near high-POI areas
     (mirroring how real charging infra tends to cluster near malls,
     highways, and business districts).

No external data source is required, so the whole project runs offline
and reproducibly (fixed random seed). Swap this module out for real
data (census, traffic-department, OpenChargeMap) without touching the
rest of the pipeline -- the rest of the code only depends on the
column names produced here.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42

# Preset city bounding boxes (lat_min, lat_max, lon_min, lon_max)
CITY_BOUNDS = {
    "Bengaluru": (12.85, 13.10, 77.50, 77.75),
    "Delhi":     (28.45, 28.75, 77.00, 77.35),
    "Mumbai":    (18.90, 19.25, 72.75, 72.98),
}


def _gaussian_hotspots(lat_grid, lon_grid, centers, rng):
    """Sum of 2D Gaussian bumps centered at `centers`, each with a
    random amplitude and spread, to emulate real-world density
    surfaces (downtown cores, business parks, etc.)."""
    surface = np.zeros_like(lat_grid, dtype=float)
    for (clat, clon) in centers:
        amplitude = rng.uniform(0.6, 1.0)
        spread = rng.uniform(0.015, 0.035)  # roughly 1.5km - 4km
        d2 = (lat_grid - clat) ** 2 + (lon_grid - clon) ** 2
        surface += amplitude * np.exp(-d2 / (2 * spread ** 2))
    return surface


def generate_city_data(city="Bengaluru", grid_size=45, n_hotspots=6,
                        n_existing_stations=25, seed=RANDOM_SEED):
    """
    Returns:
        candidates_df : one row per grid cell (candidate site)
        stations_df   : one row per existing charging station
    """
    rng = np.random.default_rng(seed)
    lat_min, lat_max, lon_min, lon_max = CITY_BOUNDS.get(city, CITY_BOUNDS["Bengaluru"])

    lats = np.linspace(lat_min, lat_max, grid_size)
    lons = np.linspace(lon_min, lon_max, grid_size)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Random "hub" centers shared by population/traffic/POI so the
    # three surfaces are correlated the way real cities are (dense
    # areas tend to have more traffic AND more shops).
    hub_centers = [
        (rng.uniform(lat_min, lat_max), rng.uniform(lon_min, lon_max))
        for _ in range(n_hotspots)
    ]

    pop_surface = _gaussian_hotspots(lat_grid, lon_grid, hub_centers, rng)
    traffic_surface = 0.7 * pop_surface + 0.3 * _gaussian_hotspots(
        lat_grid, lon_grid, hub_centers, rng
    )
    poi_surface = 0.5 * pop_surface + 0.5 * _gaussian_hotspots(
        lat_grid, lon_grid, hub_centers, rng
    )

    # Add independent noise so it isn't a perfect linear combination
    noise = lambda scale: rng.normal(0, scale, size=lat_grid.shape)
    pop_surface = np.clip(pop_surface + noise(0.05), 0, None)
    traffic_surface = np.clip(traffic_surface + noise(0.05), 0, None)
    poi_surface = np.clip(poi_surface + noise(0.05), 0, None)

    # Rescale each surface to realistic units
    population_density = 2000 + pop_surface / pop_surface.max() * 18000   # people / sq km
    traffic_volume = 500 + traffic_surface / traffic_surface.max() * 9500  # vehicles/hr (peak)
    poi_score = poi_surface / poi_surface.max() * 100                      # 0-100 commercial density index

    # Average household income index (0-100), weakly correlated with POI
    income_index = np.clip(
        35 + 0.4 * poi_score + rng.normal(0, 12, size=lat_grid.shape), 5, 100
    )

    candidates_df = pd.DataFrame({
        "site_id": [f"S{idx:04d}" for idx in range(grid_size * grid_size)],
        "latitude": lat_grid.ravel(),
        "longitude": lon_grid.ravel(),
        "population_density": population_density.ravel(),
        "traffic_volume": traffic_volume.ravel(),
        "poi_score": poi_score.ravel(),
        "income_index": income_index.ravel(),
    })

    # Existing stations: sampled with probability proportional to POI
    # score (mirrors real deployment near malls / highways / offices),
    # then snapped to nearby-but-not-identical coordinates.
    probs = candidates_df["poi_score"].to_numpy()
    probs = probs / probs.sum()
    chosen_idx = rng.choice(len(candidates_df), size=n_existing_stations,
                             replace=False, p=probs)
    stations_df = candidates_df.loc[chosen_idx, ["latitude", "longitude"]].copy()
    stations_df["latitude"] += rng.normal(0, 0.003, size=len(stations_df))
    stations_df["longitude"] += rng.normal(0, 0.003, size=len(stations_df))
    stations_df.reset_index(drop=True, inplace=True)
    stations_df["station_id"] = [f"EX{idx:03d}" for idx in range(len(stations_df))]
    stations_df["n_chargers"] = rng.integers(2, 12, size=len(stations_df))
    stations_df["avg_utilization_pct"] = rng.uniform(20, 95, size=len(stations_df)).round(1)
    stations_df = stations_df[["station_id", "latitude", "longitude",
                                "n_chargers", "avg_utilization_pct"]]

    return candidates_df, stations_df


if __name__ == "__main__":
    cand, stations = generate_city_data()
    cand.to_csv("candidate_sites.csv", index=False)
    stations.to_csv("existing_stations.csv", index=False)
    print(f"Generated {len(cand)} candidate sites and {len(stations)} existing stations.")
