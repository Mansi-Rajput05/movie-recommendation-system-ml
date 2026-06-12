"""Model training pipeline copied from main.ipynb.

The notebook trains a tuned SVD model for rating prediction and an implicit
item-based hybrid model for top-k recommendation ranking. This script preserves
those model choices and saves reusable artifacts.
"""

from pathlib import Path
import argparse
import json

import joblib
import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import train_test_split

from evaluation import evaluate_hybrid_ranking, tune_hybrid_alpha
from recommendation_generation import build_implicit_hybrid_artifacts


def train_tuned_svd(ratings_df, random_state=42):
    reader = Reader(
        rating_scale=(1, 5)
    )

    data = Dataset.load_from_df(
        ratings_df[
            ['UserID', 'MovieID', 'Rating']
        ],
        reader
    )

    trainset, testset = train_test_split(
        data,
        test_size=0.2,
        random_state=random_state
    )

    best_svd = SVD(
        n_factors=20,
        n_epochs=20,
        reg_all=0.05,
        lr_all=0.005,
        random_state=random_state
    )

    best_svd.fit(trainset)

    final_predictions = best_svd.test(testset)

    final_rmse = accuracy.rmse(
        final_predictions,
        verbose=False
    )

    final_mae = accuracy.mae(
        final_predictions,
        verbose=False
    )

    print(f"Final RMSE: {final_rmse:.4f}")
    print(f"Final MAE : {final_mae:.4f}")

    return best_svd, {
        'Final RMSE': float(final_rmse),
        'Final MAE': float(final_mae)
    }


def train_implicit_hybrid(train_df, test_df, movie_titles, random_state=42):
    hybrid_results_df, best_hybrid_row, ranking_eval_users = tune_hybrid_alpha(
        train_df,
        test_df,
        movie_titles,
        random_state=random_state
    )

    best_alpha = float(
        best_hybrid_row['alpha']
    )

    best_hybrid_map_10 = float(
        best_hybrid_row['MAP@10']
    )

    print('Best hybrid alpha:', best_alpha)
    print('Best hybrid MAP@10:', best_hybrid_map_10)

    artifacts = build_implicit_hybrid_artifacts(
        train_df,
        movie_titles,
        alpha=best_alpha
    )

    _, quality_summary = evaluate_hybrid_ranking(
        artifacts,
        test_df,
        users=ranking_eval_users,
        k=10,
        alpha=best_alpha
    )

    return artifacts, hybrid_results_df, quality_summary


def run_training(
    processed_dir="processed",
    artifacts_dir="artifacts",
    random_state=42
):
    processed_dir = Path(processed_dir)
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    ratings_df = pd.read_csv(processed_dir / "ratings_df.csv")
    train_df = pd.read_csv(processed_dir / "train_df.csv")
    test_df = pd.read_csv(processed_dir / "test_df.csv")
    movie_titles = pd.read_csv(processed_dir / "movie_titles.csv")

    best_svd, svd_metrics = train_tuned_svd(
        ratings_df,
        random_state=random_state
    )

    hybrid_artifacts, hybrid_results_df, quality_summary = train_implicit_hybrid(
        train_df,
        test_df,
        movie_titles,
        random_state=random_state
    )

    joblib.dump(
        best_svd,
        artifacts_dir / "best_svd.joblib"
    )
    joblib.dump(
        hybrid_artifacts,
        artifacts_dir / "hybrid_recommender.joblib"
    )

    hybrid_results_df.to_csv(
        artifacts_dir / "hybrid_tuning_results.csv",
        index=False
    )
    quality_summary.to_csv(
        artifacts_dir / "quality_summary.csv",
        index=False
    )

    metrics = {
        'svd': svd_metrics,
        'hybrid': {
            row['Metric']: float(row['Value'])
            for _, row in quality_summary.iterrows()
        },
        'best_alpha': float(hybrid_artifacts['alpha'])
    }

    with open(artifacts_dir / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)

    return best_svd, hybrid_artifacts, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train SVD and hybrid recommendation models."
    )
    parser.add_argument("--processed-dir", default="processed")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    run_training(
        processed_dir=args.processed_dir,
        artifacts_dir=args.artifacts_dir,
        random_state=args.random_state
    )


if __name__ == "__main__":
    main()
