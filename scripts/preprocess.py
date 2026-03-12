#!/usr/bin/env python
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
import os

from data_loader import combine_data, create_matchups
from feature_engineering import (
    weight_recent_games, agg_weight, normalize_by_opponent, 
    normalize_by_home_court, scaled_stats, create_new_stats, 
    join_matchup_stats, calculate_team_level_features
)

warnings.simplefilter(action="ignore", category=RuntimeWarning)

# Configuration
INPUT_PATH = 'input'
OUTPUT_PATH = 'output'

def process_regular_season_stats():
    """
    Orchestrates the loading, transformation, and normalization of regular season data.
    """
    print("Processing Regular Season Stats...")
    
    # 1. Load and Combine Detailed Results
    mens = pd.read_csv(f'{INPUT_PATH}/MRegularSeasonDetailedResults.csv')
    womens = pd.read_csv(f'{INPUT_PATH}/WRegularSeasonDetailedResults.csv')
    detailed_results = combine_data(mens, womens)

    # 2. Reshape data (one record per team per game)
    stat_columns = [
        'Score', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 
        'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF',
    ]
    
    stat_columns_win = ['W' + col for col in stat_columns]
    stat_columns_lose = ['L' + col for col in stat_columns]
    stat_columns_against = [col + '_against' for col in stat_columns]
    stat_features = stat_columns + stat_columns_against

    win_to_norm = dict(zip(stat_columns_win, stat_columns))
    lose_to_against = dict(zip(stat_columns_lose, stat_columns_against))
    lose_to_norm = dict(zip(stat_columns_lose, stat_columns))
    win_to_against = dict(zip(stat_columns_win, stat_columns_against))

    loc_map = {"H": 1, "N": 0, "A": -1}
    detailed_results["win_home_away"] = detailed_results["WLoc"].map(loc_map)
    detailed_results["lose_home_away"] = detailed_results["win_home_away"].multiply(-1)

    winning_team = detailed_results.rename(
        columns=win_to_norm | lose_to_against | {"WTeamID": "TeamId", "LTeamID": "TeamId_against", "win_home_away": "home_away"}
    ).drop(['WLoc', 'lose_home_away'], axis=1)
    winning_team['Win'] = 1

    losing_team = detailed_results.rename(
        columns=lose_to_norm | win_to_against | {"LTeamID": "TeamId", "WTeamID": "TeamId_against", "lose_home_away": "home_away"}
    ).drop(['WLoc', 'win_home_away'], axis=1)
    losing_team['Win'] = 0

    team_stats = pd.concat([winning_team, losing_team], ignore_index=True)
    team_stats[stat_features] = team_stats[stat_features].astype(float)
    team_stats.to_csv(f'{OUTPUT_PATH}/TeamStatsRegular.csv', index=False)

    # 3. Apply Weighting and Normalization
    weighted = weight_recent_games(team_stats, stat_features)
    normalized = normalize_by_opponent(weighted, stat_features)
    normalized = normalize_by_home_court(normalized, stat_features)
    final_agg = agg_weight(normalized, stat_features)
    
    # Calculate advanced efficiency metrics on the game level data before gathering
    team_stats_advanced = create_new_stats(team_stats)
    advanced = calculate_team_level_features(team_stats_advanced)
    final_agg = final_agg.merge(advanced, on=["League", "Season", "TeamId"])
    
    final_agg.to_csv(f'{OUTPUT_PATH}/TeamStatsRegularOppHomeNorm.csv', index=False)
    return final_agg

def process_rankings():
    """
    Processes end-of-season Massey Ordinals into average team rankings.
    """
    print("Processing Rankings...")
    ranks = pd.read_csv(f'{INPUT_PATH}/MMasseyOrdinals.csv')
    final_ranks = ranks[ranks["RankingDayNum"] == 133]
    final_ranks = final_ranks.groupby(['Season', 'TeamID'])["OrdinalRank"].mean().reset_index()
    final_ranks = final_ranks.rename(columns={"OrdinalRank": "avg_rank", "TeamID": "TeamId"})
    
    final_ranks.to_csv(f'{OUTPUT_PATH}/TeamAvgRanks.csv', index=False)
    return final_ranks

def prepare_modeling_data(stats, final_ranks):
    """
    Combines stats, ranks, and seeds into final training and testing datasets.
    """
    print("Preparing Modeling Data...")
    
    # 1. Feature Definition
    stat_columns = ['Score', 'FGper', 'FG3per', 'FTper', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']
    stat_features = stat_columns + [col + '_against' for col in stat_columns]
    eff_metrics = ["OEFF", "DEFF", "NET_EFF", "eFG", "TS", "ORper", "DRper", "TOper", "AST_TO", "3P_Reliance", "3P_Reliance_against", "3P_Defense", "FTR", "STLper", "BLKper", "Pace"]
    advanced_metrics = ["Score_Variance", "NET_EFF_Variance", "Close_Game_Win_Per", "NET_EFF_Last_10", "Opp_OEFF", "Opp_DEFF", "Conf_NET_EFF"]
    all_features = stat_features + eff_metrics + advanced_metrics + ["avg_rank", "Seed"]

    # 2. Add Efficiency Metrics
    stats = create_new_stats(stats)
    
    # 2b. Add Conference Average NET_EFF
    mens_conf = pd.read_csv(f'{INPUT_PATH}/MTeamConferences.csv')
    womens_conf = pd.read_csv(f'{INPUT_PATH}/WTeamConferences.csv')
    confs = combine_data(mens_conf, womens_conf)
    
    stats_with_conf = stats.merge(confs.rename(columns={'TeamID': 'TeamId'}), on=['League', 'Season', 'TeamId'], how='left')
    conf_avg = stats_with_conf.groupby(['League', 'Season', 'ConfAbbrev'])['NET_EFF'].mean().reset_index().rename(columns={'NET_EFF': 'Conf_NET_EFF'})
    stats = stats_with_conf.merge(conf_avg, on=['League', 'Season', 'ConfAbbrev'], how='left')
    stats.drop(columns=['ConfAbbrev'], inplace=True)
    
    
    # 3. Scaling
    stats_scaled = scaled_stats(stats, ['League', 'Season'], stat_features, StandardScaler()).fillna(0)
    ranks_scaled = scaled_stats(final_ranks, ['Season'], ["avg_rank"], MinMaxScaler()).fillna(0)
    
    combined_data = stats_scaled.merge(ranks_scaled, on=["Season", "TeamId"], how='left')

    # 4. Process Seeds
    mens_seeds = pd.read_csv(f'{INPUT_PATH}/MNCAATourneySeeds.csv')
    womens_seeds = pd.read_csv(f'{INPUT_PATH}/WNCAATourneySeeds.csv')
    seeds = combine_data(mens_seeds, womens_seeds)
    seeds['Seed'] = seeds['Seed'].str.extract(r'(\d+)').astype(int)
    
    seeds_scaled = scaled_stats(seeds, ['Season'], ["Seed"], MinMaxScaler()).fillna(0)
    seeds_processed = seeds_scaled[['Season', 'TeamID', 'Seed']].merge(
        seeds[['Season', 'TeamID', 'Seed']].rename(columns={'Seed': 'RawSeed'}), 
        on=['Season', 'TeamID']
    )
    
    combined_data = combined_data.merge(
        seeds_processed, left_on=["Season", "TeamId"], right_on=["Season", "TeamID"], how='left'
    ).drop(columns=['TeamID'])
    
    combined_data['Seed'] = combined_data['Seed'].fillna(1.0)
    combined_data['RawSeed'] = combined_data['RawSeed'].fillna(17.0)
    combined_data.to_csv(f'{OUTPUT_PATH}/CombinedSeasonStats.csv', index=False)

    # 5. Create Tournament and Regular Season Datasets
    # Tournament
    mens_t = pd.read_csv(f'{INPUT_PATH}/MNCAATourneyCompactResults.csv')
    womens_t = pd.read_csv(f'{INPUT_PATH}/WNCAATourneyCompactResults.csv')
    tourney_results = combine_data(mens_t, womens_t)
    tourney_results = tourney_results[(tourney_results["Season"] >= 2003) & (tourney_results["Season"] <= 2025)]
    tourney_matchups = create_matchups(tourney_results)
    tourney_data = join_matchup_stats(tourney_matchups, combined_data, all_features)
    tourney_data.to_csv(f'{OUTPUT_PATH}/TournamentDataModel.csv', index=False)

    # Regular Season
    mens_rs = pd.read_csv(f'{INPUT_PATH}/MRegularSeasonCompactResults.csv')
    womens_rs = pd.read_csv(f'{INPUT_PATH}/WRegularSeasonCompactResults.csv')
    rs_results = combine_data(mens_rs, womens_rs)
    rs_results = rs_results[(rs_results["Season"] >= 2003) & (rs_results["Season"] <= 2025)]
    rs_matchups = create_matchups(rs_results)
    rs_data = join_matchup_stats(rs_matchups, combined_data, all_features)
    rs_data.to_csv(f'{OUTPUT_PATH}/RegularDataModel.csv', index=False)

    # Deep Learning Combined (Optional Phase 4)
    rs_data['IsTournament'] = 0
    tourney_data['IsTournament'] = 1
    combined_all = pd.concat([rs_data, tourney_data], ignore_index=True)
    combined_all.to_csv(f'{OUTPUT_PATH}/CombinedAllGames.csv', index=False)
    
    print("Pre-processing Complete.")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        
    stats = process_regular_season_stats()
    ranks = process_rankings()
    prepare_modeling_data(stats, ranks)
