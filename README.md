# Movie Recommendation System

This repository contains a Netflix movie recommendation system. The original exploratory workflow is in `main.ipynb`, and the same core code has been extracted into reusable pipeline scripts for project submission.

## Repository Contents

- `main.ipynb`: Complete notebook with EDA, model development, evaluation, top-k recommendation examples, success cases, and observations.
- `data_preprocessing.py`: Data preprocessing pipeline. Parses Netflix raw data when required, loads movie titles, creates the filtered modeling dataset, and saves train/test splits.
- `model_training.py`: Model training pipeline. Trains the tuned SVD rating model and the final implicit item-based hybrid recommender.
- `evaluation.py`: Evaluation script. Computes ranking metrics including `Precision@10`, `Recall@10`, `Hit Rate@10`, and `MAP@10`.
- `recommendation_generation.py`: Recommendation generation module. Loads the saved hybrid recommender and generates top-k movie recommendations for a user.
- `requirements.txt`: Python dependencies needed to run the notebook and scripts.
- `combined_data_1.txt`: Netflix ratings data in original combined format.
- `netflix_ratings_1.csv`: Parsed ratings data generated from `combined_data_1.txt`.
- `movie_titles.csv`: Movie metadata used to attach titles and release years to recommendations.

## Pipeline Overview

1. Data preprocessing

The preprocessing pipeline follows the notebook logic:

- Parse `combined_data_1.txt` into columns `UserID`, `MovieID`, `Rating`, and `Date` when `netflix_ratings_1.csv` is not already present.
- Load `movie_titles.csv` with `MovieID`, `Year`, and `Title`.
- Sample 3,000,000 ratings using `random_state=42`.
- Keep users with at least 10 ratings.
- Keep movies with at least 50 ratings.
- Create an 80:20 train/test split.

2. Model training

Two model paths from the notebook are preserved:

- Tuned SVD rating model with `n_factors=20`, `n_epochs=20`, `reg_all=0.05`, `lr_all=0.005`, and `random_state=42`.
- Final implicit item-based hybrid recommender using positive interactions where `Rating >= 4`, item-item cosine similarity, popularity score, and tuned hybrid weight `alpha`.

3. Evaluation

The evaluation script measures top-k recommendation quality using held-out positive test movies:

- `Precision@10`
- `Recall@10`
- `Hit Rate@10`
- `MAP@10`

4. Recommendation generation

The recommendation module returns top-k unseen movie recommendations for a selected user and joins movie titles for readable output.

## Reproduce Results

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Run preprocessing:

```bash
python data_preprocessing.py --data-dir . --output-dir processed
```

Train models and save artifacts:

```bash
python model_training.py --processed-dir processed --artifacts-dir artifacts
```

Evaluate the saved recommender:

```bash
python evaluation.py --processed-dir processed --artifacts-dir artifacts --top-k 10
```

Generate sample recommendations for a user:

```bash
python recommendation_generation.py --artifacts-dir artifacts --user-id 305344 --top-k 10
```

## Generated Outputs

After running the scripts, the following directories are created:

- `processed/`: `df_model.csv`, `ratings_df.csv`, `train_df.csv`, `test_df.csv`, and processed `movie_titles.csv`.
- `artifacts/`: `best_svd.joblib`, `hybrid_recommender.joblib`, `hybrid_tuning_results.csv`, `quality_summary.csv`, and `metrics.json`.

## Notes

- The notebook remains the full analysis report.
- The scripts are extracted from the notebook to satisfy submission requirements for preprocessing, training, evaluation, recommendation generation, documentation, and reproducibility.
- Training can take time because the notebook uses a 3,000,000-rating sample and computes an item-item similarity matrix for the ranking model.
