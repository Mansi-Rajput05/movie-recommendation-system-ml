"""Data preprocessing pipeline copied from main.ipynb.

This script parses the Netflix combined-data text file when needed, loads movie
metadata, creates the same sampled and filtered modeling dataset used in the
notebook, and writes reproducible train/test CSV files.
"""

from pathlib import Path
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_combined_data(input_path):
    data = []
    current_movie = None

    with open(input_path, "r") as file:
        for line in file:
            line = line.strip()

            if line.endswith(":"):
                current_movie = int(line[:-1])

            else:
                user_id, rating, date = line.split(",")

                data.append([
                    int(user_id),
                    current_movie,
                    int(rating),
                    date
                ])

    df = pd.DataFrame(
        data,
        columns=["UserID", "MovieID", "Rating", "Date"]
    )

    return df


def load_or_create_ratings(data_dir):
    data_dir = Path(data_dir)
    ratings_path = data_dir / "netflix_ratings_1.csv"
    raw_path = data_dir / "combined_data_1.txt"

    if ratings_path.exists():
        df = pd.read_csv(ratings_path)
    else:
        df = parse_combined_data(raw_path)
        df.to_csv(ratings_path, index=False)

    return df


def load_movie_titles(data_dir):
    data_dir = Path(data_dir)

    movie_titles = pd.read_csv(
        data_dir / "movie_titles.csv",
        header=None,
        names=["MovieID", "Year", "Title"],
        encoding="latin-1",
        engine="python",
        on_bad_lines="skip"
    )

    return movie_titles


def create_model_dataset(
    df,
    sample_size=3000000,
    min_user_ratings=10,
    min_movie_ratings=50,
    random_state=42
):
    df_model = df.sample(
        n=min(sample_size, len(df)),
        random_state=random_state
    )

    user_counts = df_model.groupby('UserID')['Rating'].count()

    active_users = user_counts[
        user_counts >= min_user_ratings
    ].index

    df_model = df_model[
        df_model['UserID'].isin(active_users)
    ]

    movie_counts = df_model.groupby('MovieID')['Rating'].count()

    popular_movies = movie_counts[
        movie_counts >= min_movie_ratings
    ].index

    df_model = df_model[
        df_model['MovieID'].isin(popular_movies)
    ]

    ratings_df = df_model[['UserID', 'MovieID', 'Rating']].copy()

    return df_model, ratings_df


def create_train_test_split(ratings_df, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(
        ratings_df,
        test_size=test_size,
        random_state=random_state
    )

    return train_df, test_df


def run_preprocessing(
    data_dir=".",
    output_dir="processed",
    sample_size=3000000,
    min_user_ratings=10,
    min_movie_ratings=50,
    random_state=42
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_or_create_ratings(data_dir)
    movie_titles = load_movie_titles(data_dir)

    df_model, ratings_df = create_model_dataset(
        df,
        sample_size=sample_size,
        min_user_ratings=min_user_ratings,
        min_movie_ratings=min_movie_ratings,
        random_state=random_state
    )

    train_df, test_df = create_train_test_split(
        ratings_df,
        test_size=0.2,
        random_state=random_state
    )

    df_model.to_csv(output_dir / "df_model.csv", index=False)
    ratings_df.to_csv(output_dir / "ratings_df.csv", index=False)
    train_df.to_csv(output_dir / "train_df.csv", index=False)
    test_df.to_csv(output_dir / "test_df.csv", index=False)
    movie_titles.to_csv(output_dir / "movie_titles.csv", index=False)

    print("Ratings:", len(df_model))
    print("Users:", df_model['UserID'].nunique())
    print("Movies:", df_model['MovieID'].nunique())
    print("Train Ratings:", len(train_df))
    print("Test Ratings:", len(test_df))

    return df_model, ratings_df, train_df, test_df, movie_titles


def main():
    parser = argparse.ArgumentParser(
        description="Run the Netflix recommendation preprocessing pipeline."
    )
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--output-dir", default="processed")
    parser.add_argument("--sample-size", type=int, default=3000000)
    parser.add_argument("--min-user-ratings", type=int, default=10)
    parser.add_argument("--min-movie-ratings", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    run_preprocessing(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        min_user_ratings=args.min_user_ratings,
        min_movie_ratings=args.min_movie_ratings,
        random_state=args.random_state
    )


if __name__ == "__main__":
    main()
