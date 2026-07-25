"""
run_pipeline.py
----------------
Command-line entry point. Runs the full analysis + recommendation
pipeline and writes:
  outputs/candidate_sites_scored.csv
  outputs/recommended_stations.csv
  outputs/recommendation_map.html

Usage:
    python run_pipeline.py
    python run_pipeline.py --city Delhi --n_new_sites 15
"""

import argparse
import os

from src.pipeline import run_full_pipeline
from src.mapping import build_map


def main():
    parser = argparse.ArgumentParser(description="EV Charging Station Recommender")
    parser.add_argument("--city", default="Bengaluru", choices=["Bengaluru", "Delhi", "Mumbai"])
    parser.add_argument("--grid_size", type=int, default=45)
    parser.add_argument("--n_existing_stations", type=int, default=25)
    parser.add_argument("--n_new_sites", type=int, default=10)
    parser.add_argument("--min_dist_existing_km", type=float, default=1.2)
    parser.add_argument("--min_dist_between_new_km", type=float, default=1.5)
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    result = run_full_pipeline(
        city=args.city,
        grid_size=args.grid_size,
        n_existing_stations=args.n_existing_stations,
        n_new_sites=args.n_new_sites,
        min_dist_existing_km=args.min_dist_existing_km,
        min_dist_between_new_km=args.min_dist_between_new_km,
    )

    print("Model performance on held-out test split:")
    print(f"  R^2  : {result['model_metrics']['r2']}")
    print(f"  MAE  : {result['model_metrics']['mae']}")
    print(f"  Train/Test rows: {result['model_metrics']['n_train']}/{result['model_metrics']['n_test']}")
    print()
    print("Top feature importances:")
    print(result["feature_importances"].to_string(index=False))
    print()
    print(f"Selected {len(result['recommendations'])} new station sites.")

    result["candidates"].to_csv(os.path.join(args.out_dir, "candidate_sites_scored.csv"), index=False)
    result["recommendations"].to_csv(os.path.join(args.out_dir, "recommended_stations.csv"), index=False)

    fmap = build_map(
        result["candidates"], result["stations"], result["recommendations"],
        min_dist_between_new_km=args.min_dist_between_new_km,
    )
    map_path = os.path.join(args.out_dir, "recommendation_map.html")
    fmap.save(map_path)
    print(f"\nSaved scored candidates, recommendations CSV, and interactive map to '{args.out_dir}/'.")


if __name__ == "__main__":
    main()
