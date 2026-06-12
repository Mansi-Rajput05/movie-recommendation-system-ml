"""Generate figures used by TECHNICAL_REPORT.md.

The EDA figures are computed from the included Netflix ratings and movie-title
files. Model-result figures use the exact notebook outputs reported in
main.ipynb and TECHNICAL_REPORT.md.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_DIR = Path(".")
OUTPUT_DIR = Path("report_images")
RATINGS_PATH = DATA_DIR / "netflix_ratings_1.csv"
MOVIES_PATH = DATA_DIR / "movie_titles.csv"
CHUNK_SIZE = 1_000_000


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "font.size": 10,
        "grid.color": "#d9d9d9",
        "grid.linestyle": "-",
        "grid.alpha": 0.45,
        "savefig.bbox": "tight",
    })


def load_chunked_stats():
    rating_counts = pd.Series(dtype="int64")
    user_counts = pd.Series(dtype="int64")
    movie_counts = pd.Series(dtype="int64")
    movie_sums = pd.Series(dtype="float64")
    year_counts = pd.Series(dtype="int64")
    year_sums = pd.Series(dtype="float64")

    total_rows = 0

    for chunk in pd.read_csv(
        RATINGS_PATH,
        usecols=["UserID", "MovieID", "Rating", "Date"],
        chunksize=CHUNK_SIZE,
    ):
        total_rows += len(chunk)

        rating_counts = rating_counts.add(chunk["Rating"].value_counts(), fill_value=0)
        user_counts = user_counts.add(chunk["UserID"].value_counts(), fill_value=0)
        movie_counts = movie_counts.add(chunk["MovieID"].value_counts(), fill_value=0)
        movie_sums = movie_sums.add(chunk.groupby("MovieID")["Rating"].sum(), fill_value=0)

        years = chunk["Date"].astype(str).str.slice(0, 4)
        year_counts = year_counts.add(years.value_counts(), fill_value=0)
        year_sums = year_sums.add(chunk.groupby(years)["Rating"].sum(), fill_value=0)

    stats = {
        "total_rows": total_rows,
        "rating_counts": rating_counts.sort_index().astype(int),
        "user_counts": user_counts.astype(int),
        "movie_counts": movie_counts.astype(int),
        "movie_average": (movie_sums / movie_counts).sort_index(),
        "year_counts": year_counts.sort_index().astype(int),
        "year_average": (year_sums / year_counts).sort_index(),
    }

    return stats


def save_dataset_overview(stats):
    total = stats["total_rows"]
    users = len(stats["user_counts"])
    movies = len(stats["movie_counts"])
    sparsity = 1 - (total / (users * movies))

    cards = [
        ("Raw Ratings", f"{total:,}"),
        ("Unique Users", f"{users:,}"),
        ("Unique Movies", f"{movies:,}"),
        ("Observed Matrix", f"{(1 - sparsity) * 100:.3f}%"),
        ("Final Modeling Ratings", "1,968,136"),
        ("Final Modeling Movies", "2,208"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(11, 5.8))
    fig.suptitle("Netflix Rating Data Scale and Sparsity", fontsize=17, fontweight="bold")

    colors = ["#284b63", "#3c6e71", "#d9a441", "#7b2d26", "#52796f", "#b56576"]
    for ax, (label, value), color in zip(axes.ravel(), cards, colors):
        ax.set_facecolor(color)
        ax.text(0.5, 0.62, value, ha="center", va="center", color="white", fontsize=22, fontweight="bold")
        ax.text(0.5, 0.32, label, ha="center", va="center", color="white", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.text(
        0.5,
        0.02,
        "Only a tiny fraction of the possible user-movie matrix is observed, so sparse-data ranking methods are required.",
        ha="center",
        fontsize=10,
        color="#333333",
    )
    fig.savefig(OUTPUT_DIR / "dataset_overview.png", dpi=180)
    plt.close(fig)


def save_rating_distribution(stats):
    counts = stats["rating_counts"]
    proportions = counts / counts.sum()

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    bars = ax.bar(counts.index.astype(str), counts.values, color="#3c6e71")
    ax.set_title("Rating Distribution Shows a Positive Bias")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of ratings")
    ax.grid(axis="y")
    ax.ticklabel_format(axis="y", style="plain")

    for bar, pct in zip(bars, proportions):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{pct:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.savefig(OUTPUT_DIR / "rating_distribution.png", dpi=180)
    plt.close(fig)


def save_activity_distributions(stats):
    user_counts = stats["user_counts"]
    movie_counts = stats["movie_counts"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    axes[0].hist(user_counts, bins=70, color="#52796f", edgecolor="white")
    axes[0].set_title("Ratings Per User")
    axes[0].set_xlabel("Ratings per user")
    axes[0].set_ylabel("Number of users")
    axes[0].set_xscale("log")
    axes[0].grid(axis="y")

    axes[1].hist(movie_counts, bins=70, color="#b56576", edgecolor="white")
    axes[1].set_title("Ratings Per Movie")
    axes[1].set_xlabel("Ratings per movie")
    axes[1].set_ylabel("Number of movies")
    axes[1].set_xscale("log")
    axes[1].grid(axis="y")

    fig.suptitle("User Activity and Movie Popularity Are Highly Imbalanced", fontsize=15, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "activity_distributions.png", dpi=180)
    plt.close(fig)


def save_long_tail(stats):
    movie_counts = stats["movie_counts"].sort_values(ascending=False)
    cumulative = movie_counts.cumsum() / movie_counts.sum()
    x = np.arange(1, len(movie_counts) + 1) / len(movie_counts)

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.plot(x * 100, cumulative * 100, color="#7b2d26", linewidth=2.5)
    ax.axhline(80, color="#333333", linestyle="--", linewidth=1)
    idx_80 = int(np.searchsorted(cumulative.values, 0.8))
    pct_movies_80 = x[idx_80] * 100
    ax.axvline(pct_movies_80, color="#333333", linestyle="--", linewidth=1)
    ax.text(
        pct_movies_80 + 1,
        76,
        f"{pct_movies_80:.1f}% of movies\nproduce 80% of ratings",
        fontsize=10,
        color="#333333",
    )
    ax.set_title("Movie Popularity Has a Strong Long-Tail Pattern")
    ax.set_xlabel("Movies sorted by popularity (%)")
    ax.set_ylabel("Cumulative share of ratings (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True)
    fig.savefig(OUTPUT_DIR / "long_tail_popularity.png", dpi=180)
    plt.close(fig)


def save_temporal_activity(stats):
    year_counts = stats["year_counts"]
    year_average = stats["year_average"]
    year_index = year_counts.index.astype(int)

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.plot(year_index, year_counts.values, color="#284b63", linewidth=2.2, marker="o", markersize=3)
    ax1.set_title("Rating Activity Increased Strongly Over Time")
    ax1.set_xlabel("Rating year")
    ax1.set_ylabel("Number of ratings", color="#284b63")
    ax1.tick_params(axis="y", labelcolor="#284b63")
    ax1.grid(True)
    ax1.ticklabel_format(axis="y", style="plain")

    ax2 = ax1.twinx()
    ax2.plot(year_index, year_average.values, color="#d9a441", linewidth=2.0, marker="s", markersize=3)
    ax2.set_ylabel("Average rating", color="#9a6a00")
    ax2.tick_params(axis="y", labelcolor="#9a6a00")
    ax2.set_ylim(2.5, 4.2)

    fig.savefig(OUTPUT_DIR / "temporal_activity.png", dpi=180)
    plt.close(fig)


def save_popularity_vs_rating(stats):
    movie_stats = pd.DataFrame({
        "RatingCount": stats["movie_counts"],
        "AverageRating": stats["movie_average"],
    }).dropna()
    filtered = movie_stats[movie_stats["RatingCount"] >= 50].copy()
    corr = filtered["RatingCount"].corr(filtered["AverageRating"])

    plot_df = filtered.sample(n=min(2500, len(filtered)), random_state=42)

    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    ax.scatter(
        plot_df["RatingCount"],
        plot_df["AverageRating"],
        alpha=0.45,
        s=24,
        color="#3c6e71",
        edgecolor="none",
    )
    ax.set_xscale("log")
    ax.set_title("Popularity and Average Rating Are Only Weakly Related")
    ax.set_xlabel("Number of ratings per movie (log scale)")
    ax.set_ylabel("Average rating")
    ax.grid(True)
    ax.text(
        0.03,
        0.94,
        f"Correlation = {corr:.3f}",
        transform=ax.transAxes,
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#999999"},
    )
    fig.savefig(OUTPUT_DIR / "popularity_vs_average_rating.png", dpi=180)
    plt.close(fig)


def save_model_error_comparison():
    models = ["Item-Based CF", "Tuned SVD"]
    rmse = [1.0163375893800153, 0.929426]
    mae = [0.8074920395950065, 0.733191]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    b1 = ax.bar(x - width / 2, rmse, width, label="RMSE", color="#284b63")
    b2 = ax.bar(x + width / 2, mae, width, label="MAE", color="#d9a441")
    ax.set_title("Tuned SVD Improves Rating Prediction Accuracy")
    ax.set_ylabel("Error (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y")

    for bars in [b1, b2]:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.018,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    fig.savefig(OUTPUT_DIR / "model_error_comparison.png", dpi=180)
    plt.close(fig)


def save_topk_comparison():
    models = ["Tuned SVD\nRanking", "Popularity\nBaseline", "Implicit Hybrid\nRecommender"]
    map_values = [0.00818, 0.023459542076089697, 0.03080411664357101]
    colors = ["#7b2d26", "#d9a441", "#3c6e71"]

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    bars = ax.bar(models, map_values, color=colors)
    ax.set_title("Ranking-Focused Hybrid Model Has the Best MAP@10")
    ax.set_ylabel("MAP@10 (higher is better)")
    ax.set_ylim(0, 0.036)
    ax.grid(axis="y")

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{bar.get_height():.5f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.savefig(OUTPUT_DIR / "topk_model_comparison.png", dpi=180)
    plt.close(fig)


def save_hybrid_alpha_tuning():
    alpha = np.array([0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00])
    map_10 = np.array([0.029409, 0.029579, 0.029956, 0.030171, 0.030229, 0.030715, 0.030785, 0.030804, 0.030681, 0.030752])

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.plot(alpha, map_10, marker="o", linewidth=2.4, color="#3c6e71")
    best_idx = int(np.argmax(map_10))
    ax.scatter(alpha[best_idx], map_10[best_idx], s=120, color="#d9a441", zorder=3, edgecolor="#333333")
    ax.annotate(
        "Best alpha = 0.50\nMAP@10 = 0.03080",
        xy=(alpha[best_idx], map_10[best_idx]),
        xytext=(0.55, 0.03025),
        arrowprops={"arrowstyle": "->", "color": "#333333"},
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#999999"},
    )
    ax.set_title("Hybrid Weight Tuning Shows Similarity Adds Ranking Value")
    ax.set_xlabel("alpha: weight on item similarity")
    ax.set_ylabel("MAP@10")
    ax.set_ylim(0.0292, 0.0310)
    ax.grid(True)
    fig.savefig(OUTPUT_DIR / "hybrid_alpha_tuning.png", dpi=180)
    plt.close(fig)


def save_ranking_quality_summary():
    metrics = ["Precision@10", "Recall@10", "Hit Rate@10", "MAP@10"]
    values = [0.02160, 0.09582, 0.19767, 0.03080]
    colors = ["#284b63", "#52796f", "#3c6e71", "#d9a441"]

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    bars = ax.bar(metrics, values, color=colors)
    ax.set_title("Final Hybrid Recommender Quality at Top 10")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 0.23)
    ax.grid(axis="y")

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.006,
            f"{bar.get_height():.5f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.savefig(OUTPUT_DIR / "hybrid_quality_summary.png", dpi=180)
    plt.close(fig)


def save_sample_recommendations():
    movies = [
        "The Sixth Sense",
        "The Silence of the Lambs",
        "Pirates of the Caribbean",
        "LOTR: Fellowship",
        "Braveheart",
        "Lethal Weapon",
        "Finding Nemo",
        "Jaws",
        "Shrek 2",
        "The Wizard of Oz",
    ]
    scores = [0.9843, 0.9715, 0.9436, 0.9397, 0.9280, 0.9169, 0.9085, 0.9080, 0.8979, 0.8885]

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    y = np.arange(len(movies))[::-1]
    bars = ax.barh(y, scores, color="#3c6e71")
    ax.set_yticks(y)
    ax.set_yticklabels(movies)
    ax.set_xlabel("Hybrid score")
    ax.set_title("Example Top-10 Recommendations for User 305344")
    ax.set_xlim(0.84, 1.00)
    ax.grid(axis="x")

    for bar in bars:
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.4f}",
            va="center",
            fontsize=9,
        )

    fig.savefig(OUTPUT_DIR / "sample_recommendations.png", dpi=180)
    plt.close(fig)


def save_dashboard(source_files, output_file, title, nrows, ncols):
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 7.2))
    axes = np.asarray(axes).ravel()

    for ax, source_file in zip(axes, source_files):
        image = plt.imread(OUTPUT_DIR / source_file)
        ax.imshow(image)
        ax.axis("off")

    for ax in axes[len(source_files):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTPUT_DIR / output_file, dpi=180)
    plt.close(fig)


def main():
    apply_style()
    OUTPUT_DIR.mkdir(exist_ok=True)

    stats = load_chunked_stats()
    save_dataset_overview(stats)
    save_rating_distribution(stats)
    save_activity_distributions(stats)
    save_long_tail(stats)
    save_temporal_activity(stats)
    save_popularity_vs_rating(stats)
    save_model_error_comparison()
    save_topk_comparison()
    save_hybrid_alpha_tuning()
    save_ranking_quality_summary()
    save_sample_recommendations()
    save_dashboard(
        [
            "rating_distribution.png",
            "long_tail_popularity.png",
            "popularity_vs_average_rating.png",
        ],
        "eda_summary_dashboard.png",
        "EDA Summary: Rating Bias, Long Tail, and Popularity vs Quality",
        1,
        3,
    )
    save_dashboard(
        [
            "model_error_comparison.png",
            "hybrid_alpha_tuning.png",
            "topk_model_comparison.png",
        ],
        "model_results_dashboard.png",
        "Model Results: Rating Accuracy, Hybrid Tuning, and Ranking Quality",
        1,
        3,
    )

    print(f"Saved report figures to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
