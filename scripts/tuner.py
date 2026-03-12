from sklearn.metrics import make_scorer, brier_score_loss
from skopt import BayesSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np

def tune_models(season_data, features):
    """
    Uses BayesSearchCV to optimize model hyperparameters for XGBoost, Random Forest, and Logistic Regression.
    
    :param season_data: Training data.
    :param features: List of features to use.
    :return: Dictionary of best estimators for each model type.
    """
    brier_scorer = make_scorer(brier_score_loss, greater_is_better=False, response_method='predict_proba')

    X_train = season_data[features]
    y_train = season_data['Pred']
    
    models_to_tune = {
        'xgb': {
            'model': XGBClassifier(random_state=42),
            'params': {
                'n_estimators': (50, 300),
                'learning_rate': (0.01, 0.3, 'log-uniform'),
                'max_depth': (3, 8),
                'subsample': (0.5, 1.0),
                'colsample_bytree': (0.5, 1.0)
            }
        },
        'rf': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': (50, 300),
                'max_depth': (3, 12),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 20)
            }
        },
        'lr': {
            'model': LogisticRegression(solver='liblinear', random_state=42, max_iter=1000),
            'params': {
                'C': (1e-3, 1e2, 'log-uniform'),
                'penalty': ['l1', 'l2']
            }
        }
    }
    
    best_estimators = {}
    
    for name, config in models_to_tune.items():
        print(f"\nTuning {name}...")
        bayes_search = BayesSearchCV(
            estimator=config['model'],
            search_spaces=config['params'],
            n_iter=20, 
            cv=3, 
            scoring=brier_scorer,
            n_jobs=-1,
            random_state=42
        )
        
        bayes_search.fit(X_train, y_train)
        
        print(f"Best Parameters for {name}:", bayes_search.best_params_)
        print(f"Best CV Brier Score for {name}:", -bayes_search.best_score_)
        
        best_estimators[name] = bayes_search.best_estimator_
        
    return best_estimators
