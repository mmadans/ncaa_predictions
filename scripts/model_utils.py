import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import brier_score_loss


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
