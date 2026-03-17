#!/usr/bin/env python
import pandas as pd 
import numpy as np 
import warnings
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from predictor import final_predictions

warnings.simplefilter(action="ignore", category=RuntimeWarning)

# Configuration
INPUT_PATH = 'input'
OUTPUT_PATH = 'output'

def train_mens_model(tourney_data, rs_data):
    """
    Trains and calibrates the Men's league model using RS + Tourney data.
    """
    print("\n--- Training Men's Model ---")
    
    # 1. Dataset Preparation
    # Training: All RS (up to 2026) + All Tourney (up to 2024)
    # Validation: 2025 Tourney
    rs_history_m = rs_data[rs_data['League'] == 'M'] # Includes 2026
    tourney_history_m = tourney_data[(tourney_data['League'] == 'M') & (tourney_data['Season'] < 2025)]
    train_data_m = pd.concat([rs_history_m, tourney_history_m], ignore_index=True).fillna(0)
    
    tourney_data_m_val = tourney_data[(tourney_data['League'] == 'M') & (tourney_data['Season'] == 2025)].fillna(0)
    
    # Features from FINAL_MODEL_CONFIG.md
    features = [
        'Seed', 'Seed_Diff_Squared', 'Score', 'Score_against', 'FGper', 'FG3per', 'FTper', 
        'FGper_against', 'FG3per_against', 'FTper_against', 'OEFF', 'DEFF', 'NET_EFF', 
        'eFG', 'TS', 'ORper', 'DRper', 'TOper', 'AST_TO', '3P_Reliance', '3P_Reliance_against', 
        '3P_Defense', 'FTR', 'STLper', 'Pace', 'Score_Variance', 'NET_EFF_Variance', 
        'Close_Game_Win_Per', 'NET_EFF_Last_10', 'Opp_OEFF', 'Opp_DEFF', 'Conf_NET_EFF', 'avg_rank'
    ]

    X_train = train_data_m[features]
    y_train = train_data_m['Pred']
    X_val = tourney_data_m_val[features]
    y_val = tourney_data_m_val['Pred']

    # 4. Ensemble and Calibration (Fixed Weights)
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)
    lr = LogisticRegression(solver='liblinear', random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    models = [('xgb', xgb), ('lr', lr), ('rf', rf)]
    ensemble = VotingClassifier(estimators=models, voting='soft', weights=[2, 1, 1])
    ensemble.fit(X_train, y_train)

    calibrated_ensemble = CalibratedClassifierCV(ensemble, method='isotonic', cv=3)
    calibrated_ensemble.fit(X_train, y_train)

    # 5. Evaluation (on 2025 Tournament)
    probs = calibrated_ensemble.predict_proba(X_val)[:, 1]
    score = brier_score_loss(y_val, probs)
    print(f"Men's Calibrated Ensemble Brier Score (2025 Validation): {score:.4f}")

    return calibrated_ensemble, features

def train_womens_model(tourney_data, rs_data):
    """
    Trains and calibrates the Women's league model using Tourney-only data.
    """
    print("\n--- Training Women's Model ---")
    
    # 1. Dataset Preparation
    # Training: Tourney (up to 2024) + RS (including 2026 - joining if needed, but per original logic it used tourney-only)
    # Let's add RS data for Women as well to follow the user request for 2026 historical data inclusion
    rs_history_w = rs_data[rs_data['League'] == 'W'] # Includes 2026
    tourney_history_w = tourney_data[(tourney_data['League'] == 'W') & (tourney_data['Season'] < 2025)]
    train_data_w = pd.concat([rs_history_w, tourney_history_w], ignore_index=True).fillna(0)
    
    tourney_data_w_val = tourney_data[(tourney_data['League'] == 'W') & (tourney_data['Season'] == 2025)].fillna(0)
    
    # Features from FINAL_MODEL_CONFIG.md
    features = [
        'Seed', 'Seed_Diff_Squared', 'Score', 'Score_against', 'FGper', 'FG3per', 'FTper', 
        'FGper_against', 'FTper_against', 'DEFF', 'DRper', 'TOper', 'AST_TO', '3P_Reliance', 
        'FTR', 'STLper', 'Pace', 'Score_Variance', 'Close_Game_Win_Per', 'NET_EFF_Last_10', 'Conf_NET_EFF'
    ]

    X_train = train_data_w[features]
    y_train = train_data_w['Pred']
    X_val = tourney_data_w_val[features]
    y_val = tourney_data_w_val['Pred']

    # 4. Ensemble and Calibration (Fixed Weights)
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)
    lr = LogisticRegression(solver='liblinear', random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    models = [('xgb', xgb), ('lr', lr), ('rf', rf)]
    ensemble = VotingClassifier(estimators=models, voting='soft', weights=[1, 2, 1]) 
    ensemble.fit(X_train, y_train)

    calibrated_ensemble = CalibratedClassifierCV(ensemble, method='isotonic', cv=3)
    calibrated_ensemble.fit(X_train, y_train)

    # 5. Evaluation (on 2025 Tournament)
    probs = calibrated_ensemble.predict_proba(X_val)[:, 1]
    score = brier_score_loss(y_val, probs)
    print(f"Women's Calibrated Ensemble Brier Score (2025 Validation): {score:.4f}")

    return calibrated_ensemble, features

def main():
    """
    Main orchestration for training, evaluation, and prediction generation.
    """
    # 1. Load Data
    tourney_data = pd.read_csv(f'{OUTPUT_PATH}/TournamentDataModel.csv')
    rs_data = pd.read_csv(f'{OUTPUT_PATH}/RegularDataModel.csv')
    combined_stats = pd.read_csv(f'{OUTPUT_PATH}/CombinedSeasonStats.csv')

    # 2. Train Models
    model_m, features_m = train_mens_model(tourney_data, rs_data)
    model_w, features_w = train_womens_model(tourney_data, rs_data)

    # 3. Generate Predictions (Final for all 2026 D1 teams)
    print("\nGenerating Predictions for All Possible 2026 Matchups...")
    teams_m = pd.read_csv(f'{INPUT_PATH}/MTeams.csv')
    teams_m = teams_m[teams_m["LastD1Season"] == 2026] 
    pred_m = final_predictions(teams_m, combined_stats.fillna(0), 2026, features_m, model_m, correction=0.05, boost_1_seeds=True, boost_high_conf=True, round_extremes=False)
    pred_m.to_csv(f'{OUTPUT_PATH}/final_predictions_m.csv', index=False)

    teams_w = pd.read_csv(f'{INPUT_PATH}/WTeams.csv')
    active_w_teams = combined_stats[(combined_stats["League"] == "W") & (combined_stats["Season"] == 2026)]["TeamId"].unique()
    teams_w = teams_w[teams_w["TeamID"].isin(active_w_teams)]
    pred_w = final_predictions(teams_w, combined_stats.fillna(0), 2026, features_w, model_w, correction=0, boost_1_seeds=True, boost_high_conf=True, round_extremes=False)
    pred_w.to_csv(f'{OUTPUT_PATH}/final_predictions_w.csv', index=False)

    # 4. Final Submission
    submission = pd.concat([pred_m, pred_w])[['ID', 'Pred']]
    submission.to_csv(f'{OUTPUT_PATH}/submission.csv', index=False)
    print(f"Refactored Submission generation complete. Saved to {OUTPUT_PATH}/submission.csv")

if __name__ == "__main__":
    main()
