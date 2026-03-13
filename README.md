# NCAA March Madness 2026 Prediction Model

This repository contains the finalized 2026 NCAA March Madness prediction model for both Men's and Women's leagues. The model is an ensemble of `XGBoost`, `RandomForest`, and `LogisticRegression`, calibrated with Isotonic Regression.

The project is designed to be a streamlined production pipeline, with all developmental, experimental, and mathematical logic preserved in dedicated Jupyter notebooks for transparency and future iteration.

---

## 🚀 Quick Start (2026 Execution)

To generate the final submission when the 2026 tournament data is available:

1.  **Setup Environment**:
    ```bash
    uv sync
    ```

2.  **Run Data Engineering**:
    Place new datasets in `input/` and run:
    ```bash
    uv run python scripts/preprocess.py
    ```

3.  **Train & Predict**:
    Train the finalized ensembles and generate `output/submission.csv`:
    ```bash
    uv run python scripts/train.py
    ```

---

## 📁 Project Structure

### 🐍 Production Scripts (`scripts/`)
These scripts are optimized for the final 2026 run and use the validated configurations from `FINAL_MODEL_CONFIG.md`.

*   **[`preprocess.py`](file:///Users/michael/Documents/Data%20Projects/ncaa_predictions/scripts/preprocess.py)**: Orchestrates data loading, normalization, and modeling dataset creation.
*   **[`train.py`](file:///Users/michael/Documents/Data%20Projects/ncaa_predictions/scripts/train.py)**: Trains the calibrated voting ensembles using fixed features and hyperparameters.
*   **[`feature_engineering.py`](file:///Users/michael/Documents/Data%20Projects/ncaa_predictions/scripts/feature_engineering.py)**: Core logic for recency weighting, opponent adjustment, and home-court normalization.
*   **[`predictor.py`](file:///Users/michael/Documents/Data%20Projects/ncaa_predictions/scripts/predictor.py)**: Generates win probabilities for all possible matchups with post-prediction adjustments (e.g., 1-seed boosting).

### 📓 Development Notebooks (`notebooks/`)
Experimental logic and historical developmental processes are documented here:
*   **[`feature_engineering.ipynb`](file:///Users/michael/Documents/Data%20Projects/ncaa_predictions/notebooks/feature_engineering.ipynb)**: Detailed mathematical breakdown of the normalization and weighting logic.
*   **[`model_comparisons.ipynb`](file:///Users/michael/Documents/Data%20Projects/ncaa_predictions/notebooks/model_comparisons.ipynb)**: Feature importance analysis (L1 coefficients) and Brier Score comparisons across different architectures.
*   **[`hyper_parameter_tuning.ipynb`](file:///Users/michael/Documents/Data%20Projects/ncaa_predictions/notebooks/hyper_parameter_tuning.ipynb)**: Bayesian optimization logic used to find the final model parameters.

### 📄 Configuration
*   **[`FINAL_MODEL_CONFIG.md`](file:///Users/michael/Documents/Data%20Projects/ncaa_predictions/FINAL_MODEL_CONFIG.md)**: The authoritative source for feature lists, model weights, and post-prediction overrides.

---

## 🧠 Model Architecture

### 1. Feature Engineering
- **Recency Weighting**: Linearly emphasizes games later in the season ($Weight = 1 + \frac{DayNum}{MaxDayNum}$).
- **Normalization**: Adjusted for opponent defensive/offensive strength and removed home-court variance.
- **Advanced Metrics**: Offensive/Defensive Efficiency (points per possession), eFG%, and net efficiency trends.

### 2. Modeling Strategy
We use a **Soft Voting Ensemble** of three base models:
- **XGBoost**: Captures non-linearities and high-variance upsets.
- **Logistic Regression**: Provides stability and baseline linear relationships.
- **Random Forest**: Aggregates variance and reduces overfitting.

**Calibration**: All ensemble outputs are transformed using `CalibratedClassifierCV` (Isotonic Regression) to ensure probabilities are reliable for Brier Score optimization.

### 3. League-Specific Adjustments
- **Men's Model**: Weighted towards XGBoost (2:1:1) to handle the higher frequency of large-scale upsets.
- **Women's Model**: Weighted towards Logistic Regression (1:2:1) due to the higher historical predictability of top seeds.

---

## 📊 2025 Holdout Performance
| League | Brier Score |
| :--- | :--- |
| **Men's** | **0.1509** |
| **Women's** | **0.1650** |

---

## 🔎 Future Improvements
- Integrate coaching track records and historical "upset-prone" profiles.
- Incorporate player-level injury data and transfer portal impacts.
- Explore deeper neural network architectures (MLP/Transformer) using the data preparation logic in `notebooks/feature_engineering.ipynb`.
