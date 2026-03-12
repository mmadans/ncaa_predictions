# NCAA March Madness 2025 Men's and Women's Predictions

This project was developed for the annual NCAA March Madness competition hosted on [Kaggle](https://www.kaggle.com/competitions/march-machine-learning-mania-2025/overview). The model predicts win probabilities for all possible Division I matchups across both Men's and Women's leagues, evaluated based on real outcomes from the 2025 March Madness tournament. Submissions are scored using Brier Score (the same as Mean Squared Error in this context).


## 📂 Input

This repository includes all datasets provided by the competition. While not all datasets were used in the final model, they are included for potential future improvements.

## 📄 Output

The `submission.csv` file contains the final predictions submitted to the competition. It includes two columns:

- `ID`: A concatenation of the season and team IDs for any given matchup (lower TeamID first). Example: "2024\_1101\_1234"
- `Pred`: A predicted win probability between 0 and 1 for Team 1.


## 📘 Project Structure

The project has been refactored into a modular Python-based pipeline for better maintainability and clarity.

### 📁 scripts/
This directory contains the core logic of the prediction pipeline:

*   **`preprocess.py`**: The primary entry point for data preparation. It orchestrates the entire processing flow, from raw CSV ingestion to feature-ready modeling datasets.
*   **`train.py`**: The main modeling script. It handles league-specific training (Combined RS+Tourney for Men, Tourney-only for Women), model tuning, ensemble creation, and final prediction generation.
*   **`data_loader.py`**: Contains utility functions for loading, cleaning, and combining raw NCAA datasets.
*   **`feature_engineering.py`**: Houses the logic for complex transformations, including weighted averaging, multi-level normalization (opponent and home-court), and efficient rate calculation.
*   **`model_utils.py`**: Provides helper functions for L1-based feature selection and model performance evaluation (Brier Score analysis).
*   **`tuner.py`**: Encapsulates the hyperparameter optimization logic using `BayesSearchCV`.
*   **`predictor.py`**: Manages the generation of win probabilities and the formatting of competition-compliant outputs.

### 📁 notebooks/
Contains exploratory Jupyter notebooks (`.ipynb`) used for initial analysis and visualization.

---

## 🚀 Execution Order

To reproduce the model and generate a new submission, follow these steps in order:

1.  **Environment Setup**:
    Ensure dependencies are installed (managed via `uv` or `pip`).
    ```bash
    uv sync
    ```

2.  **Data Pre-processing**:
    Run the pre-processing script to generate normalized team stats and matchup datasets.
    ```bash
    uv run python scripts/preprocess.py
    ```
    *Output*: `output/CombinedSeasonStats.csv`, `output/TournamentDataModel.csv`, `output/RegularDataModel.csv`.

3.  **Model Training & Prediction**:
    Run the training script to optimize models and generate the final win probabilities.
    ```bash
    uv run python scripts/train.py
    ```
    *Output*: `output/submission.csv` and specific league prediction files.

---

## 🔧 Methodology

### Feature Engineering

Key steps to construct the feature set for training and evaluation:

- Weighted games to emphasize those later in the season.
- Normalized game stats based on opponent strength and home-court advantage.
- Created new efficiency metrics from normalized stats (e.g. Offensive/Defensive Efficiency, eFG%, Pace).
- Incorporated team-level momentum and variance metrics (e.g. NET_EFF Variance, Last 10 Games NET_EFF, Close Game Win Percentage).
- Proxied Strength of Schedule (SOS) and Conference Strength by aggregating opponent and conference-level efficiencies.
- Incorporated end-of-season team ranks for the Men's league (data not available for women's).
- Standardized all features (Z-scaling for game stats, Min-Max scaling for ranks).
- Selected features dynamically using L1 Regularization to minimize Brier Score.
- Merged engineered features with historical matchups to create final training and testing datasets.

Using 2025 regular season data, the model predicts 2025 tournament outcomes. Historical regular season data serves as training, while historical tournament outcomes are used for testing.

- **Men's Model:** `XGBoost` provided the best performance.
- **Women's Model:** `LogisticRegression` from `scikit-learn` performed best.

Steps to optimize model performance:

1. **Feature Selection:** Used chi-squared tests to select significant features, minimizing Brier Score.
2. **Hyperparameter Tuning:** Applied `BayesSearchCV` from `scikit-optimize` to refine model parameters.
3. **Final Prediction:** Trained the optimized model to predict outcomes for all potential 2025 matchups.

---

## 📊 Results

The model's initial tournament Brier Score was **0.15852**, which placed around the 50th percentile on the leaderboard. Following post-tournament review, several advanced features were integrated to better capture team dynamics (Pace, SOS, Variance, Recency, 3PT Defense). 

With these final adjustments evaluated on the 2025 hold-out data:
- **Men's Model:** Improved cross-validated Brier Score to **0.1534**
- **Women's Model:** Improved cross-validated Brier Score to **0.1763**

- **Context:** In 2023, the winning model had a slightly higher Brier Score, but performance typically declines in later rounds when matchups are more competitive. The new feature set provides a substantial edge.
- **Observation:** The final iteration model successfully identifies teams with strong recent momentum and high variance, correcting to give less conservative win probabilities for potential upsets compared to the initial model.

---

## 🔎 Next Steps

To improve the model for future competitions:

- Integrate official tournament seeding alongside end-of-season rankings.
- Incorporate coaching experience and track record.
- Analyze macro-level trends across seasons to identify years more prone to upsets and adjust predictions accordingly.

I had a great experience building this model and plan to refine it for future March Madness competitions!
