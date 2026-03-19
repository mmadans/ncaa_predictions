# 2026 NCAA March Madness Final Model Configuration

This document saves the finalized model configurations for both the Men's and Women's NCAA predictions. When the 2026 data is uploaded, we will use these exact settings to train and generate the final predictions.

## 🏀 Men's Model Configuration

**Holdout 2025 Brier Score:** 0.1509

### 1. Architecture
- **Base Models:** Voting Ensemble (Soft Voting)
  - `XGBClassifier` (Weight: 2)
  - `LogisticRegression` (Weight: 1)
  - `RandomForestClassifier` (Weight: 1)
- **Calibration:** `CalibratedClassifierCV` using Isotonic Regression (cv=3)

### 2. Selected Features (L1 Regularized)
`Seed`, `Seed_Diff_Squared`, `Score`, `Score_against`, `FGper`, `FG3per`, `FTper`, `FGper_against`, `FG3per_against`, `FTper_against`, `OEFF`, `DEFF`, `NET_EFF`, `eFG`, `TS`, `ORper`, `DRper`, `TOper`, `AST_TO`, `3P_Reliance`, `3P_Reliance_against`, `3P_Defense`, `FTR`, `STLper`, `Pace`, `Score_Variance`, `NET_EFF_Variance`, `Close_Game_Win_Per`, `NET_EFF_Last_10`, `Opp_OEFF`, `Opp_DEFF`, `Conf_NET_EFF`, `avg_rank`

### 3. Post-Prediction Adjustments
- `boost_1_seeds=True`: 1 vs 16 seeds boosted by 5%.
- `boost_high_conf=True`: Any prediction >85% is boosted by 2.5%.
- `correction=0.05`: Any prediction extremely close to 0 or 1 (<5% or >95%) is rounded to 0 or 1 respectively.
- `round_extremes=False`

---

## 🏀 Women's Model Configuration

**Holdout 2025 Brier Score:** 0.1650

### 1. Architecture
- **Base Models:** Voting Ensemble (Soft Voting)
  - `LogisticRegression` (Weight: 2) -> *Heavily Weighted for Women's*
  - `XGBClassifier` (Weight: 1)
  - `RandomForestClassifier` (Weight: 1)
- **Calibration:** `CalibratedClassifierCV` using Isotonic Regression (cv=3)

### 2. Selected Features (L1 Regularized)
`Seed`, `Seed_Diff_Squared`, `Score`, `Score_against`, `FGper`, `FG3per`, `FTper`, `FGper_against`, `FTper_against`, `DEFF`, `DRper`, `TOper`, `AST_TO`, `3P_Reliance`, `FTR`, `STLper`, `Pace`, `Score_Variance`, `Close_Game_Win_Per`, `NET_EFF_Last_10`, `Conf_NET_EFF`

### 3. Post-Prediction Adjustments
- `boost_1_seeds=True`: 1 vs 16 seeds boosted by 5%.
- `boost_high_conf=True`: Any prediction >85% is boosted by 2.5%.
- `correction=0`: No general correction rounding applied.
- `round_extremes=False`

---

## 🚀 2026 Run Instructions

When the new 2026 tournament datasets are uploaded from Kaggle, simply follow these steps:

1. Place the new datasets into the `input/` folder (replacing the old ones).
2. Run data engineering: `uv run python scripts/preprocess.py`
3. Train the calibrated ensembled models and output predictions: `uv run python scripts/train.py`
4. Submit `output/submission.csv`!
