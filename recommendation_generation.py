"""Recommendation generation module copied from main.ipynb.

The final notebook model is an implicit item-based hybrid recommender. It uses
positive interactions (`Rating >= 4`), item-item cosine similarity, and a
popularity prior to produce top-k movie recommendations.
"""

from pathlib import Path
import argparse

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def build_implicit_hybrid_artifacts(trainset_df, movie_titles, alpha=0.5):
    positive_train = trainset_df[
        trainset_df['Rating'] >= 4
    ].copy()

    all_rank_movies = np.array(
        sorted(trainset_df['MovieID'].unique())
    )

    movie_to_idx = {
        movie_id: idx
        for idx, movie_id in enumerate(all_rank_movies)
    }

    positive_users = np.array(
        sorted(positive_train['UserID'].unique())
    )

    user_to_idx = {
        user_id: idx
        for idx, user_id in enumerate(positive_users)
    }

    positive_train['UserIndex'] = positive_train['UserID'].map(user_to_idx)
    positive_train['MovieIndex'] = positive_train['MovieID'].map(movie_to_idx)

    positive_matrix = csr_matrix(
        (
            np.ones(len(positive_train), dtype=np.float32),
            (
                positive_train['UserIndex'].to_numpy(),
                positive_train['MovieIndex'].to_numpy()
            )
        ),
        shape=(
            len(positive_users),
            len(all_rank_movies)
        )
    )

    item_popularity = np.asarray(
        positive_matrix.sum(axis=0)
    ).ravel().astype(np.float32)

    popularity_score = np.log1p(item_popularity)

    if popularity_score.max() > 0:
        popularity_score = popularity_score / popularity_score.max()

    item_similarity = (
        positive_matrix.T @ positive_matrix
    ).astype(np.float32).toarray()

    item_norms = np.sqrt(
        np.diag(item_similarity)
    )

    denominator = np.outer(
        item_norms,
        item_norms
    )

    item_similarity = np.divide(
        item_similarity,
        denominator,
        out=np.zeros_like(item_similarity),
        where=denominator != 0
    )

    np.fill_diagonal(
        item_similarity,
        0
    )

    positive_movies_by_user = (
        positive_train
        .groupby('UserID')['MovieID']
        .apply(set)
        .to_dict()
    )

    rated_movies_by_user = (
        trainset_df
        .groupby('UserID')['MovieID']
        .apply(set)
        .to_dict()
    )

    return {
        'alpha': alpha,
        'movie_titles': movie_titles,
        'all_rank_movies': all_rank_movies,
        'movie_to_idx': movie_to_idx,
        'popularity_score': popularity_score,
        'item_similarity': item_similarity,
        'positive_movies_by_user': positive_movies_by_user,
        'rated_movies_by_user': rated_movies_by_user,
        'trainset_df': trainset_df
    }


def score_implicit_hybrid(artifacts, user_id, alpha=None):
    if alpha is None:
        alpha = artifacts['alpha']

    scores = (
        1 - alpha
    ) * artifacts['popularity_score'].copy()

    liked_movies = artifacts['positive_movies_by_user'].get(
        user_id,
        set()
    )

    liked_indices = [
        artifacts['movie_to_idx'][movie_id]
        for movie_id in liked_movies
        if movie_id in artifacts['movie_to_idx']
    ]

    if liked_indices:
        similarity_score = artifacts['item_similarity'][:, liked_indices].mean(axis=1)

        if similarity_score.max() > 0:
            similarity_score = similarity_score / similarity_score.max()

        scores += alpha * similarity_score

    rated_movies = artifacts['rated_movies_by_user'].get(
        user_id,
        set()
    )

    rated_indices = [
        artifacts['movie_to_idx'][movie_id]
        for movie_id in rated_movies
        if movie_id in artifacts['movie_to_idx']
    ]

    if rated_indices:
        scores[rated_indices] = -np.inf

    return scores


def recommend_implicit_hybrid(artifacts, user_id, n=10, alpha=None):
    scores = score_implicit_hybrid(
        artifacts,
        user_id,
        alpha=alpha
    )

    top_indices = np.argpartition(
        scores,
        -n
    )[-n:]

    top_indices = top_indices[
        np.argsort(scores[top_indices])[::-1]
    ]

    return artifacts['all_rank_movies'][top_indices].tolist()


def recommend_movies_implicit_hybrid(artifacts, user_id, n=10, alpha=None):
    scores = score_implicit_hybrid(
        artifacts,
        user_id,
        alpha=alpha
    )

    top_indices = np.argpartition(
        scores,
        -n
    )[-n:]

    top_indices = top_indices[
        np.argsort(scores[top_indices])[::-1]
    ]

    recommendations = pd.DataFrame({
        'MovieID': artifacts['all_rank_movies'][top_indices],
        'Hybrid_Score': scores[top_indices]
    })

    recommendations = recommendations.merge(
        artifacts['movie_titles'],
        on='MovieID',
        how='left'
    )

    recommendations.insert(
        0,
        'Rank',
        range(1, len(recommendations) + 1)
    )

    return recommendations


def get_user_profile(artifacts, user_id, max_movies=8):
    profile = (
        artifacts['trainset_df'][artifacts['trainset_df']['UserID'] == user_id]
        .sort_values('Rating', ascending=False)
        .head(max_movies)
        .merge(artifacts['movie_titles'], on='MovieID', how='left')
    )

    return profile[[
        'MovieID',
        'Title',
        'Year',
        'Rating'
    ]]


def main():
    parser = argparse.ArgumentParser(
        description="Generate top-k movie recommendations for a user."
    )
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    artifacts_path = Path(args.artifacts_dir) / "hybrid_recommender.joblib"
    artifacts = joblib.load(artifacts_path)

    print("User profile from training data:")
    print(get_user_profile(artifacts, args.user_id))
    print()
    print(f"Top-{args.top_k} recommendations:")
    print(recommend_movies_implicit_hybrid(
        artifacts,
        args.user_id,
        n=args.top_k
    ))


if __name__ == "__main__":
    main()
