import pandas as pd
import numpy as np
from itertools import product

def final_predictions(teams, stats, season, features, model, correction=0, boost_1_seeds=False, boost_high_conf=False, round_extremes=False):
    """
    Generates win probabilities for all possible team matchups for a given season.
    
    :param teams: Dataframe of teams.
    :param stats: Team stats for the season.
    :param season: The target season year.
    :param features: Features used by the model.
    :param model: Trained model.
    :param correction: Probability rounding threshold.
    :param boost_high_conf: Boost probabilities >80% by 5%.
    :param round_extremes: Round probabilities >95% to 1.0 and <5% to 0.0.
    :return: Submission-ready dataframe with 'ID' and 'Pred'.
    """
    team_list = teams['TeamID'].unique()
    team_combos = pd.DataFrame(product(team_list, team_list), columns=['team1', 'team2'])
    team_combos = team_combos[team_combos["team1"] != team_combos["team2"]]
    
    team_combos["TeamID_first"] = team_combos[['team1', 'team2']].min(axis=1)
    team_combos["TeamID_second"] = team_combos[['team1', 'team2']].max(axis=1)
    team_combos["Season"] = season
    team_combos['ID'] = team_combos['Season'].astype('str') + '_' + team_combos['TeamID_first'].astype('str') + '_' + team_combos['TeamID_second'].astype('str') 
    team_combos = team_combos[["TeamID_first", "TeamID_second", "Season", "ID"]].drop_duplicates()
    
    season_stats = stats[stats["Season"] == season]
    team_stats = team_combos.merge(
        season_stats, left_on=['Season', 'TeamID_first'], right_on=['Season', 'TeamId'], how='inner'
    ).merge(
        season_stats, left_on=['Season', 'TeamID_second'], right_on=['Season', 'TeamId'], how='inner', suffixes=('_first', '_second')
    )
    
    for feat in features:
        if feat == 'Seed_Diff_Squared':
            continue
        team_stats[feat] = team_stats[feat+'_first'] - team_stats[feat+'_second']
        
    if 'Seed_Diff_Squared' in features:
        if 'RawSeed_first' in team_stats.columns:
            seed_diff_raw = team_stats['RawSeed_first'] - team_stats['RawSeed_second']
            team_stats['Seed_Diff_Squared'] = (seed_diff_raw ** 2) * np.sign(seed_diff_raw)
        else:
            team_stats['Seed_Diff_Squared'] = np.sign(team_stats['Seed']) * (team_stats['Seed'] ** 2)
    
    predictions = model.predict_proba(team_stats[features])[:, 1]
    
    team_stats["Pred"] = predictions
    
    if boost_1_seeds:
        # Boost probabilities for 1 vs 16 seed matchups by 10%
        # First team is 1 seed, second is 16 seed (boost Pred by 0.10)
        team_stats.loc[(team_stats['RawSeed_first'] == 1) & (team_stats['RawSeed_second'] == 16), "Pred"] += 0.10
        # First team is 16 seed, second is 1 seed (boost Pred by -0.10)
        team_stats.loc[(team_stats['RawSeed_first'] == 16) & (team_stats['RawSeed_second'] == 1), "Pred"] -= 0.10
        
        # Ensure probabilities remain bounded between 0 and 1
        team_stats["Pred"] = team_stats["Pred"].clip(0.0, 1.0)
        
    if boost_high_conf:
        # Boost confidence for >80% implicitly favoring the favorite. 
        # But we must clip values!
        team_stats.loc[team_stats["Pred"] > 0.80, "Pred"] += 0.05
        team_stats.loc[team_stats["Pred"] < 0.20, "Pred"] -= 0.05
        team_stats["Pred"] = team_stats["Pred"].clip(0.0, 1.0)
        
    if round_extremes:
        team_stats.loc[team_stats["Pred"] > 0.95, "Pred"] = 1.0
        team_stats.loc[team_stats["Pred"] < 0.05, "Pred"] = 0.0
        
    if correction > 0:
        team_stats["Pred"] = team_stats["Pred"].apply(lambda x: round(x) if x <= correction or 1 - correction <= x else x)
    
    return team_stats[["ID", "Pred"]]
