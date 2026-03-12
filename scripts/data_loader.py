import pandas as pd
import numpy as np

def combine_data(mens_data, womens_data):
    """
    Combines men's and women's datasets into one combined output.
    
    :param mens_data: Men's dataset.
    :param womens_data: Women's dataset.
    :return: Combined dataset with a 'League' column.
    """
    mens_data['League'] = 'M'
    womens_data['League'] = 'W'
    combined = pd.concat([mens_data, womens_data], axis=0)
    return combined

def create_matchups(df): 
    """
    Creates competition-compliant IDs for each matchup and determines game winners.
    
    :param df: Input game-level data.
    :return: Game results with 'ID' (e.g., Season_Team1_Team2) and 'Pred' (1 if Team1 won).
    """
    df['TeamID_first'] = df[['WTeamID', 'LTeamID']].min(axis=1)
    df['TeamID_second'] = df[['WTeamID', 'LTeamID']].max(axis=1)
    
    df['ID'] = df['Season'].astype('str') + '_' + df['TeamID_first'].astype('str') + '_' + df['TeamID_second'].astype('str') 
   
    df["Pred"] = np.where(df['WTeamID'] < df['LTeamID'], 1, 0)
    
    df = df[["TeamID_first", "TeamID_second", "League", "Season", "ID", "Pred"]]
    
    return df
