import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def weight_recent_games(df, stat_columns):
    """
    Applies linear weighting to game stats to emphasize games later in the season.
    
    :param df: Input game-level data.
    :param stat_columns: List of target stats to weight.
    :return: Dataframe with weighted stats.
    """
    output = df.copy()
    output['Weight'] = 1 + (output['DayNum'] / output.groupby(['League', 'Season'])['DayNum'].transform('max'))

    for col in stat_columns:
        output[col] = output[col] * output['Weight']

    return output

def agg_weight(df, stat_columns):
    """
    Aggregates game data to the season/team level using a weighted average.

    :param df: Input game-level data with 'Weight' column.
    :param stat_columns: List of stats to aggregate.
    :return: Weighted averages for each stat at the Season and Team level.
    """
    season_agg = df.groupby(['League', 'Season', 'TeamId']).apply(
        lambda x: (x[stat_columns].sum() / x['Weight'].sum()),
        include_groups=False
    ).reset_index()

    return season_agg

def normalize_by_opponent(df, stat_columns):
    """
    Adjusts team stats based on the defensive/offensive strength of their opponents.
    
    :param df: Input game-level data.
    :param stat_columns: List of stats to normalize.
    :return: Normalized game stats.
    """
    data_agg = agg_weight(df, stat_columns)
    
    opp_stats = df.merge(
        data_agg.rename(columns={'TeamId':'TeamId_against'} | {x:f"opp_{x}" for x in stat_columns}),
        on=["League", "Season", "TeamId_against"],
        how='left'
    )

    for col in stat_columns:
        if "_against" in col:
            opp_stats[col] = opp_stats[col] / opp_stats[f'opp_{col}']
        else:
            opp_stats[col] = opp_stats[col] / opp_stats[f'opp_{col}_against']

    output = opp_stats.drop(columns=[f"opp_{x}" for x in stat_columns])

    return output

def normalize_by_home_court(df, stat_columns):
    """
    Adjusts stats to remove the variance caused by home-court advantage.
    
    :param df: Input game-level data with 'home_away' indicator.
    :param stat_columns: List of stats to normalize.
    :return: Home-court normalized stats.
    """
    group_by = ["League", "Season", "TeamId"]
    home_data = df[df['home_away']==1]
    away_data = df[df['home_away']==-1]

    home_data_agg = agg_weight(home_data, stat_columns).rename({stat:stat + "_home" for stat in stat_columns}, axis=1)
    away_data_agg = agg_weight(away_data, stat_columns).rename({stat:stat + "_away" for stat in stat_columns}, axis=1)

    output = df.merge(home_data_agg, on=group_by).merge(away_data_agg, on=group_by)

    for stat in stat_columns:
        effect = (output[f'{stat}_home'] -  output[f'{stat}_away']) / 2  
        output.loc[output['home_away'] == 1, stat] -= effect
        output.loc[output['home_away'] == -1, stat] += effect
        
    output = output[["League", "Season", "TeamId", "DayNum", "Weight"] + stat_columns]
    
    return output

def scaled_stats(df, group_by, stat_columns, scaler):
    """
    Scales features on a per-season or per-league basis.
    
    :param df: Input data.
    :param group_by: Dimensions to group by before scaling.
    :param stat_columns: Stats to scale.
    :param scaler: Scikit-learn scaler instance.
    :return: Dataframe with scaled features.
    """
    data_scaled = df.groupby(group_by)[stat_columns].apply(
        lambda x: pd.DataFrame(
            scaler.fit_transform(x),
            columns=stat_columns,
        ),
        include_groups=False
    ).reset_index()
    
    output = df.copy()    
    output[stat_columns] = data_scaled[stat_columns]

    return output

def create_new_stats(df):
    """
    Calculates advanced efficiency and rate stats (OEFF, DEFF, eFG, etc.).
    
    :param df: Season or game level stats.
    :return: Dataframe with calculated advanced metrics.
    """
    output = df.copy()

    # Shooting
    output["FGper"] = output["FGM"] / output["FGA"]
    output["FG3per"] = output["FGM3"] / output["FGA3"]
    output["FTper"] = output["FTM"] / output["FTA"]

    output["FGper_against"] = output["FGM_against"] / output["FGA_against"]
    output["FG3per_against"] = output["FGM3_against"] / output["FGA3_against"]
    output["FTper_against"] = output["FTM_against"] / output["FTA_against"]

    # Efficiency
    output["Possessions"] = output["FGA"] + 0.44 * output["FTA"] - output["OR"] + output["TO"]
    output["Possessions_against"] = output["FGA_against"] + 0.44 * output["FTA_against"] - output["OR_against"] + output["TO_against"]

    output["OEFF"] = output["Score"] / output["Possessions"]
    output["DEFF"] = output["Score_against"] / output["Possessions_against"]
    output["NET_EFF"] = output["OEFF"] - output["DEFF"]
    
    output["eFG"] = (output["FGM"] + 0.5 * output["FGM3"]) / output["FGA"]
    output["TS"] = output["Score"] / (2 * (output["FGA"] + 0.44 * output["FTA"]))

    # Rates
    output["ORper"] = output["OR"] / (output["OR"] + output["DR_against"])
    output["DRper"] = output["DR"] / (output["DR"] + output["OR_against"]) 
    output["TOper"] = output["TO"] / output["Possessions"]
    output["AST_TO"] = output["Ast"] / output["TO"]
    output["3P_Reliance"] = output["FGA3"] / output["FGA"]
    output["3P_Reliance_against"] = output["FGA3_against"] / output["FGA_against"]
    output["3P_Defense"] = output["FG3per_against"]
    output["FTR"] = output["FTA"] / output["FGA"]
    output["STLper"] = output["Stl"] / output["Possessions"]
    output["BLKper"] = output["Blk"] / output["FGA"]
    output["Pace"] = output["Possessions"]
    
    return output

def calculate_team_level_features(df):
    """
    Calculates team-level variance and close game performance metrics from game-level data.
    """
    # Needs a copy since we add columns
    temp_df = df.copy()
    
    grouped = temp_df.groupby(["League", "Season", "TeamId"])
    
    # Game Score & NET_EFF Variance
    variance = grouped.agg(
        Score_Variance=('Score', 'std'),
        NET_EFF_Variance=('NET_EFF', 'std')
    ).reset_index()
    
    # Close Game Win Percentage
    temp_df['Score_Diff'] = np.abs(temp_df['Score'] - temp_df['Score_against'])
    close_games = temp_df[temp_df['Score_Diff'] <= 5]
    close_win_per = close_games.groupby(["League", "Season", "TeamId"])['Win'].mean().reset_index().rename(columns={'Win': 'Close_Game_Win_Per'})
    
    # Last 10 games NET_EFF 
    # Sort by DayNum to get the most recent games
    temp_df = temp_df.sort_values(by=["League", "Season", "TeamId", "DayNum"], ascending=[True, True, True, False])
    last_10 = temp_df.groupby(["League", "Season", "TeamId"]).head(10)
    recency = last_10.groupby(["League", "Season", "TeamId"]).agg(
        NET_EFF_Last_10=('NET_EFF', 'mean')
    ).reset_index()
    
    # SOS (Strength of Schedule) -> average of opponents' season-long OEFF and DEFF.
    # We will just take the average OEFF and DEFF of the teams they played against in the game-level data.
    # Note: A true SOS requires knowing the opponent's average for the season, but computing the average of OEFF_against 
    # and DEFF_against from the game stats serves as a proxy for the defensive/offensive strength of the schedule.
    # An opponent's offensive efficiency is simply the team's defensive efficiency (DEFF), and vice-versa.
    sos = grouped.agg(
        Opp_OEFF=('DEFF', 'mean'), # opponents' offensive efficiency in those games
        Opp_DEFF=('OEFF', 'mean')  # opponents' defensive efficiency in those games
    ).reset_index()

    # Matchup against base
    base = grouped.size().reset_index()[["League", "Season", "TeamId"]]
    
    result = base.merge(variance, on=["League", "Season", "TeamId"], how="left")
    result = result.merge(close_win_per, on=["League", "Season", "TeamId"], how="left")
    result = result.merge(recency, on=["League", "Season", "TeamId"], how="left")
    result = result.merge(sos, on=["League", "Season", "TeamId"], how="left")
    
    # Fill NAs 
    result['Close_Game_Win_Per'] = result['Close_Game_Win_Per'].fillna(0.5) 
    result['Score_Variance'] = result['Score_Variance'].fillna(0)
    result['NET_EFF_Variance'] = result['NET_EFF_Variance'].fillna(0)
    result['NET_EFF_Last_10'] = result['NET_EFF_Last_10'].fillna(0)
    
    return result

def join_matchup_stats(matchup, stats, feature_names):
    """
    Joins season/team stats onto a matchup dataset and calculates feature differences.
    
    :param matchup: Matchup IDs and result.
    :param stats: Team stats by season.
    :param feature_names: Features to calculate differences for.
    :return: Training/Testing dataset.
    """
    combined = matchup.merge(
        stats,
        left_on=["League", 'Season', 'TeamID_first'],
        right_on=["League", 'Season', 'TeamId'],
        how='left'
    ).merge(
        stats,
        left_on=["League", 'Season', 'TeamID_second'],
        right_on=["League", 'Season', 'TeamId'],
        how='left',
        suffixes=('_first', '_second')
    )

    for feat in feature_names:
        combined[feat] = combined[feat+'_first'] - combined[feat+'_second']
        
    if 'Seed' in feature_names:
        if 'RawSeed_first' in combined.columns:
            seed_diff_raw = combined['RawSeed_first'] - combined['RawSeed_second']
            combined['Seed_Diff_Squared'] = (seed_diff_raw ** 2) * np.sign(combined['Seed'])
        else:
            combined['Seed_Diff_Squared'] = np.sign(combined['Seed']) * (combined['Seed'] ** 2)

    output = combined[["League", "Season", "ID", "Pred"] + feature_names + (['Seed_Diff_Squared'] if 'Seed' in feature_names else [])]

    return output
