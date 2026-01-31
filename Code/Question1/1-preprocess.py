"""
Data Preprocessing for MCM Problem C Question 1
Process DWTS data to extract weekly contestant information, judge scores, and elimination results.
"""

import pandas as pd
import numpy as np
import pickle
import re
from pathlib import Path

# Configuration
DATA_PATH = Path('e:/Competition/数学建模/26美赛/Data/2026_MCM_Problem_C_Data.csv')
OUTPUT_DIR = Path('e:/Competition/数学建模/26美赛')

def load_and_clean_data():
    """Load CSV and handle N/A values"""
    df = pd.read_csv(DATA_PATH)
    
    # Replace N/A with NaN, then with 0 for scores
    df = df.replace('N/A', np.nan)
    
    # Convert score columns to numeric
    score_cols = [col for col in df.columns if 'judge' in col and 'score' in col]
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df

def parse_elimination_week(result_str):
    """Extract elimination week from results string"""
    if pd.isna(result_str):
        return None
    
    # Pattern: "Eliminated Week X"
    match = re.search(r'Eliminated Week (\d+)', str(result_str))
    if match:
        return int(match.group(1))
    
    # Special cases
    if 'Withdrew' in str(result_str):
        return None  # Handle separately
    
    return None  # For winners (1st, 2nd, 3rd place)

def get_season_weeks(df, season):
    """Get the number of weeks for a season"""
    season_df = df[df['season'] == season]
    
    # Find max week with non-zero scores
    max_week = 0
    for week in range(1, 12):
        week_cols = [f'week{week}_judge{j}_score' for j in range(1, 5)]
        if any(col in df.columns for col in week_cols):
            # Check if any contestant has non-zero scores
            scores = season_df[week_cols].sum(axis=1)
            if (scores > 0).any():
                max_week = week
    
    return max_week

def get_weekly_scores(df, season, week):
    """Extract judge scores for a specific week"""
    season_df = df[df['season'] == season].copy()
    
    week_cols = [f'week{week}_judge{j}_score' for j in range(1, 5) if f'week{week}_judge{j}_score' in df.columns]
    
    if not week_cols:
        return None
    
    # Calculate total score for the week
    season_df['judge_total'] = season_df[week_cols].sum(axis=1)
    
    # Keep only contestants with non-zero scores (still in competition)
    active_df = season_df[season_df['judge_total'] > 0].copy()
    
    if len(active_df) == 0:
        return None
    
    return active_df[['celebrity_name', 'ballroom_partner', 'celebrity_industry', 
                      'celebrity_age_during_season', 'results', 'placement', 
                      'judge_total'] + week_cols]

def get_weekly_eliminated(df, season, week):
    """Get contestants eliminated in a specific week"""
    season_df = df[df['season'] == season]
    
    eliminated = []
    for _, row in season_df.iterrows():
        elim_week = parse_elimination_week(row['results'])
        if elim_week == week:
            eliminated.append(row['celebrity_name'])
    
    return eliminated

def preprocess_all_data():
    """Main preprocessing function"""
    print("Loading data...")
    df = load_and_clean_data()
    
    # Get all seasons
    seasons = sorted(df['season'].unique())
    print(f"Found seasons: {seasons}")
    
    all_weekly_data = []
    
    for season in seasons:
        print(f"\nProcessing Season {season}...")
        
        # Determine method based on season
        if season in [1, 2]:
            method = 'rank'
        elif 3 <= season <= 27:
            method = 'percentage'
        else:  # season >= 28
            method = 'rank'  # Assumption based on problem statement
        
        # Get number of weeks for this season
        num_weeks = get_season_weeks(df, season)
        print(f"  Weeks: {num_weeks}, Method: {method}")
        
        for week in range(1, num_weeks + 1):
            weekly_scores = get_weekly_scores(df, season, week)
            
            if weekly_scores is None or len(weekly_scores) == 0:
                continue
            
            # Get eliminated contestants
            eliminated = get_weekly_eliminated(df, season, week)
            
            # Calculate judge percentages
            total_judge_score = weekly_scores['judge_total'].sum()
            weekly_scores['judge_percent'] = weekly_scores['judge_total'] / total_judge_score if total_judge_score > 0 else 0
            
            # Calculate judge ranks (1 = highest score)
            weekly_scores['judge_rank'] = weekly_scores['judge_total'].rank(ascending=False, method='min').astype(int)
            
            # Mark eliminated contestants
            weekly_scores['eliminated'] = weekly_scores['celebrity_name'].isin(eliminated)
            
            # Add metadata
            weekly_scores['season'] = season
            weekly_scores['week'] = week
            weekly_scores['method'] = method
            weekly_scores['num_contestants'] = len(weekly_scores)
            
            all_weekly_data.append(weekly_scores)
            
            if eliminated:
                print(f"    Week {week}: {len(weekly_scores)} contestants, eliminated: {eliminated}")
    
    # Combine all data
    all_data = pd.concat(all_weekly_data, ignore_index=True)
    
    print(f"\nTotal weekly records: {len(all_weekly_data)}")
    print(f"Total rows: {len(all_data)}")
    
    return all_data

def save_preprocessed_data(data):
    """Save preprocessed data"""
    # Save as pickle for Python use
    output_path = OUTPUT_DIR / '1-preprocessed_data.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved to {output_path}")
    
    # Save summary as CSV
    summary = data.groupby(['season', 'week']).agg({
        'celebrity_name': 'count',
        'eliminated': 'sum',
        'method': 'first',
        'judge_total': 'mean'
    }).reset_index()
    summary.columns = ['season', 'week', 'num_contestants', 'num_eliminated', 'method', 'avg_judge_score']
    
    summary_path = OUTPUT_DIR / '1-weekly-summary.csv'
    summary.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")
    
    return summary

if __name__ == '__main__':
    print("="*60)
    print("MCM Problem C - Question 1: Data Preprocessing")
    print("="*60)
    
    # Preprocess data
    data = preprocess_all_data()
    
    # Save results
    summary = save_preprocessed_data(data)
    
    print("\n" + "="*60)
    print("Preprocessing Complete!")
    print("="*60)
    print(f"\nSummary of processed data:")
    print(summary.head(20))
