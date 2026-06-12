"""Evaluation scripts copied from main.ipynb.

This module contains the ranking metrics used in the notebook, including
MAP@10, Precision@10, Recall@10, and Hit Rate@10.
"""

from pathlib import Path
import argparse

import joblib
import numpy as np
import pandas as pd

from recommendation_generation import (
    build_implicit_hybrid_artifacts,
    recommend_implicit_hybrid,
)


def average_precision_at_k(actual, predicted, k=10):

    predicted = predicted[:k]

    score = 0
    hits = 0

    for i, movie in enumerate(predicted):

        if movie in actual:

            hits += 1

            score += hits / (i + 1)

    if len(actual) == 0:
        return 0

    return score / min(
        len(actual),
        k
    )


def precision_at_k(actual, predicted, k=10):
    predicted = predicted[:k]

    if len(predicted) == 0:
        return 0

    return len(set(actual).intersection(predicted)) / len(predicted)


def recall_at_k(actual, predicted, k=10):
    if len(actual) == 0:
        return 0

    predicted = predicted[:k]

    return len(set(actual).intersection(predicted)) / len(actual)


def evaluate_hybrid_ranking(artifacts, testset_df, users=None, k=10, alpha=None):
    test_positive = testset_df[
        testset_df['Rating'] >= 4
    ].copy()

    actual_movies_by_user = (
        test_positive
        .groupby('UserID')['MovieID']
        .apply(set)
        .to_dict()
    )

    if users is None:
        users = np.array(list(actual_movies_by_user.keys()))

    quality_rows = []

    for user_id in users:
        actual_movies = actual_movies_by_user.get(
            user_id,
            set()
        )

        if not actual_movies:
            continue

        predicted_movies = recommend_implicit_hybrid(
            artifacts,
            user_id,
            n=k,
            alpha=alpha
        )

        hits = len(
            set(actual_movies).intersection(predicted_movies)
        )

        quality_rows.append({
            'UserID': user_id,
            'Relevant_Test_Movies': len(actual_movies),
            'Hits@10': hits,
            'Precision@10': precision_at_k(actual_movies, predicted_movies, k=k),
            'Recall@10': recall_at_k(actual_movies, predicted_movies, k=k),
            'AP@10': average_precision_at_k(
                actual_movies,
                predicted_movies,
                k=k
            )
        })

    recommendation_quality = pd.DataFrame(quality_rows)

    quality_summary = pd.DataFrame({
        'Metric': [
            'Precision@10',
            'Recall@10',
            'Hit Rate@10',
            'MAP@10'
        ],
        'Value': [
            recommendation_quality['Precision@10'].mean(),
            recommendation_quality['Recall@10'].mean(),
            (recommendation_quality['Hits@10'] > 0).mean(),
            recommendation_quality['AP@10'].mean()
        ]
    })

    return recommendation_quality, quality_summary


def tune_hybrid_alpha(
    trainset_df,
    testset_df,
    movie_titles,
    alpha_values=None,
    n_eval_users=3000,
    random_state=42
):
    if alpha_values is None:
        alpha_values = [
            0.0,
            0.02,
            0.05,
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.70,
            1.00
        ]

    artifacts = build_implicit_hybrid_artifacts(
        trainset_df,
        movie_titles,
        alpha=0.3
    )

    test_positive = testset_df[
        testset_df['Rating'] >= 4
    ].copy()

    actual_movies_by_user = (
        test_positive
        .groupby('UserID')['MovieID']
        .apply(set)
        .to_dict()
    )

    rng = np.random.default_rng(random_state)
    available_eval_users = np.array(
        list(actual_movies_by_user.keys())
    )

    n_eval_users = min(
        n_eval_users,
        len(available_eval_users)
    )

    ranking_eval_users = rng.choice(
        available_eval_users,
        size=n_eval_users,
        replace=False
    )

    hybrid_results = []

    for alpha in alpha_values:
        _, quality_summary = evaluate_hybrid_ranking(
            artifacts,
            testset_df,
            users=ranking_eval_users,
            k=10,
            alpha=alpha
        )

        map_10 = float(
            quality_summary.loc[
                quality_summary['Metric'] == 'MAP@10',
                'Value'
            ].iloc[0]
        )

        hybrid_results.append({
            'alpha': alpha,
            'MAP@10': map_10
        })

        print(
            f'alpha={alpha:.2f} | MAP@10={map_10:.5f}'
        )

    hybrid_results_df = pd.DataFrame(hybrid_results)
    best_hybrid_row = (
        hybrid_results_df
        .sort_values('MAP@10', ascending=False)
        .iloc[0]
    )

    return hybrid_results_df, best_hybrid_row, ranking_eval_users


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the saved hybrid recommender."
    )
    parser.add_argument("--processed-dir", default="processed")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    artifacts_dir = Path(args.artifacts_dir)

    test_df = pd.read_csv(processed_dir / "test_df.csv")
    artifacts = joblib.load(artifacts_dir / "hybrid_recommender.joblib")

    recommendation_quality, quality_summary = evaluate_hybrid_ranking(
        artifacts,
        test_df,
        k=args.top_k
    )

    recommendation_quality.to_csv(
        artifacts_dir / "recommendation_quality.csv",
        index=False
    )
    quality_summary.to_csv(
        artifacts_dir / "quality_summary.csv",
        index=False
    )

    print(quality_summary)


if __name__ == "__main__":
    main()
