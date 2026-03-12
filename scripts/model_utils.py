import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import brier_score_loss

def feature_select_stats(season_data, tourney_data, features, model):
    """
    Uses L1-regularized Logistic Regression (Lasso) to select the most robust features.
    
    :param season_data: Training data.
    :param tourney_data: Test data for baseline evaluation.
    :param features: List of potential features.
    :param model: Baseline model to evaluate selected features.
    :return: List of selected feature names.
    """
    X_train = season_data[features].fillna(0)
    y_train = season_data['Pred']

    print(f"Selecting features with L1 Regularization...")
    selector = LogisticRegressionCV(
        cv=5, 
        penalty='l1', 
        solver='liblinear', 
        scoring='neg_brier_score', 
        random_state=42,
        max_iter=1000
    )
    selector.fit(X_train, y_train)
    
    coefficients = selector.coef_[0]
    best_feature_set = [features[i] for i in range(len(features)) if coefficients[i] != 0]

    if len(best_feature_set) < 5:
        print("L1 suppressed too many features. Falling back to top coefficients by magnitude.")
        top_indices = sorted(range(len(coefficients)), key=lambda i: abs(coefficients[i]), reverse=True)[:10]
        best_feature_set = [features[i] for i in top_indices]

    print("Best Features: ", best_feature_set)

    # Baseline evaluation
    X_test_stat = tourney_data[best_feature_set].fillna(0)
    y_test_stat = tourney_data['Pred']

    model.fit(X_train[best_feature_set], y_train)
    y_pred_proba = model.predict_proba(X_test_stat)[:, 1]
    brier_score = brier_score_loss(y_test_stat, y_pred_proba)
    print(f"Baseline Brier Score with selected features: {brier_score:.4f}")

    return best_feature_set

def evaluate_model(prob, test):
    """
    Evaluates Brier scores at various probability rounding thresholds.
    
    :param prob: Predicted probabilities.
    :param test: Actual outcomes.
    """
    adj_prob = [0, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5]
    for adj in adj_prob:
        prob_adj = [round(x) if x <= adj or 1 - adj <= x else x for x in prob]
        brier_score = brier_score_loss(test, prob_adj) 
        print(f"Brier score (Adj={adj}): {brier_score:.4f}")
