# NCAA March Madness 2025 Men's and Women's Predictions

This project was developed for the annual NCAA March Madness competition hosted on [Kaggle](https://www.kaggle.com/competitions/march-machine-learning-mania-2025/overview). The model predicts win probabilities for all possible Division I matchups across both Men's and Women's leagues, evaluated based on real outcomes from the 2025 March Madness tournament. Submissions are scored using Brier Score (the same as Mean Squared Error in this context).


## 📂 Input

This repository includes all datasets provided by the competition. While not all datasets were used in the final model, they are included for potential future improvements.

## 📄 Output

The `submission.csv` file contains the final predictions submitted to the competition. It includes two columns:

- `ID`: A concatenation of the season and team IDs for any given matchup (lower TeamID first). Example: "2024\_1101\_1234"
- `Pred`: A predicted win probability between 0 and 1 for Team 1.


## 📘 Notebooks

The repository contains two main notebooks that handle data processing, model training, and evaluation. These are adapted from my original Kaggle notebook, accessible [here](https://www.kaggle.com/code/michaelmadans/march-madness-predictions). Moving forward, I plan to use this repo to optimize the model pipeline for future competitions.

1. **data_processing**

   - Reads and transforms competition data.
   - Performs feature engineering, including normalization, scaling, and generating efficiency metrics.
   - Prepares the final training dataset from regular season data and test dataset from tournament results.

2. **build_model**

   - Trains and evaluates separate models for Men's and Women's tournaments.
   - Conducts feature selection, hyperparameter tuning, and final output generation.

---

## 🔧 Methodology

### Feature Engineering

Key steps to construct the feature set for training and evaluation:

- Weighted games to emphasize those later in the season.
- Normalized game stats based on opponent strength and home-court advantage.
- Created new efficiency metrics from normalized stats.
- Incorporated end-of-season team ranks for the Men's league (data not avaialble for women's).
- Standardized all features (Z-scaling for game stats, Min-Max scaling for ranks).
- Merged engineered features with historical matchups to create final training and testing datasets.

### Model Creation and Evaluation

Using 2025 regular season data, the model predicts 2025 tournament outcomes. Historical regular season data serves as training, while historical tournament outcomes are used for testing.

- **Men's Model:** `XGBoost` provided the best performance.
- **Women's Model:** `LogisticRegression` from `scikit-learn` performed best.

Steps to optimize model performance:

1. **Feature Selection:** Used chi-squared tests to select significant features, minimizing Brier Score.
2. **Hyperparameter Tuning:** Applied `BayesSearchCV` from `scikit-optimize` to refine model parameters.
3. **Final Prediction:** Trained the optimized model to predict outcomes for all potential 2025 matchups.

---

## 📊 Results

At the conclusion of the tournment, the model's Brier Score is **0.15852**. This result is around the 50th percentile on hte leaderboard.

- **Context:** In 2023, the winning model had a slightly higher Brier Score, but performance typically declines in later rounds when matchups are more competitive.
- **Observation:** My model tends to give conservative win probabilities for higher-seeded teams. In a tournament with fewer early upsets, and a very chalk 2nd weekend (only the second time in history all four #1 seeds made the Final Four), this approach underperformed compared to more aggressive predictions. 

---

## 🔎 Next Steps

To improve the model for future competitions:

- Integrate official tournament seeding alongside end-of-season rankings.
- Incorporate coaching experience and track record.
- Analyze macro-level trends across seasons to identify years more prone to upsets and adjust predictions accordingly.

I had a great experience building this model and plan to refine it for future March Madness competitions!
