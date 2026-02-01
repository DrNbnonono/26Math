"""
Question 2 Data Preparation
Load Question 1 results and raw data, prepare unified dataset for method comparison
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path('e:/Competition/数学建模/26美赛')
DATA_DIR = BASE_DIR / 'Data'
Q1_DIR = DATA_DIR / 'Question1'
Q2_DIR = DATA_DIR / 'Question2'
Q2_DIR.mkdir(exist_ok=True)

def load_question1_results():
    """Load Question 1 fan vote estimates"""
    print("Loading Question 1 results...")
    
    # Load final vote estimates (combined percentage + rank methods)
    final_votes = pd.read_csv(Q1_DIR / '1-final-vote-estimates.csv')
    
    print(f"  Loaded {len(final_votes)} contestant-week records")
    print(f"  Seasons: {sorted(final_votes['season'].unique())}")
    print(f"  Methods: {final_votes['method'].unique()}")
    
    return final_votes

def load_raw_data():
    """Load raw contestant data"""
    print("\nLoading raw contestant data...")
    
    df = pd.read_csv(DATA_DIR / '2026_MCM_Problem_C_Data.csv')
    
    # Extract basic features
    contestants = df[['celebrity_name', 'ballroom_partner', 'celebrity_industry', 
                      'celebrity_homestate', 'celebrity_homecountry/region',
                      'celebrity_age_during_season', 'season', 'results', 
                      'placement']].copy()
    
    # Rename columns for consistency
    contestants.columns = ['celebrity_name', 'ballroom_partner', 'industry',
                           'home_state', 'home_country', 'age', 'season', 
                           'results', 'final_placement']
    
    # Remove duplicates (one row per contestant)
    contestants = contestants.drop_duplicates(subset=['season', 'celebrity_name'])
    
    print(f"  Loaded {len(contestants)} unique contestants")
    
    return contestants

def create_unified_dataset(final_votes, contestants):
    """Merge vote estimates with contestant features"""
    print("\nCreating unified dataset...")
    
    # Merge on season and celebrity_name
    unified = final_votes.merge(
        contestants,
        on=['season', 'celebrity_name', 'ballroom_partner'],
        how='left'
    )
    
    # Add derived features
    unified['age_group'] = pd.cut(unified['age'], 
                                   bins=[0, 30, 40, 50, 100],
                                   labels=['<30', '30-40', '40-50', '50+'])
    
    # Parse results to get elimination week
    def get_elimination_week(results_str):
        if pd.isna(results_str) or results_str == 'Winner':
            return None
        try:
            # Extract week number from strings like "Week 5"
            if 'Week' in str(results_str):
                return int(str(results_str).split('Week')[1].strip().split()[0])
        except:
            pass
        return None
    
    unified['elimination_week'] = unified['results'].apply(get_elimination_week)
    
    # Calculate survival weeks (max week for each contestant)
    survival = unified.groupby(['season', 'celebrity_name'])['week'].max().reset_index()
    survival.columns = ['season', 'celebrity_name', 'survival_weeks']
    unified = unified.merge(survival, on=['season', 'celebrity_name'], how='left')
    
    print(f"  Unified dataset: {len(unified)} records")
    print(f"  Features: {unified.columns.tolist()}")
    
    return unified

def extract_contestant_features(contestants):
    """Extract and save contestant-level features"""
    print("\nExtracting contestant features...")
    
    features = contestants.copy()
    
    # Clean industry field
    features['industry_clean'] = features['industry'].fillna('Unknown')
    
    # Combine home state and country
    features['home_clean'] = features['home_state'].fillna('') + ', ' + features['home_country'].fillna('')
    features['home_clean'] = features['home_clean'].str.strip(', ').replace('', 'Unknown')
    
    # Parse final placement
    def parse_placement(placement):
        if pd.isna(placement):
            return None
        try:
            return int(placement)
        except:
            return None
    
    features['placement_numeric'] = features['final_placement'].apply(parse_placement)
    
    print(f"  Extracted features for {len(features)} contestants")
    print(f"  Industries: {features['industry_clean'].nunique()}")
    print(f"  Dancers: {features['ballroom_partner'].nunique()}")
    
    return features

def save_outputs(unified, features):
    """Save preprocessed data"""
    print("\nSaving outputs...")
    
    # Save unified dataset as pickle (preserves dtypes)
    with open(Q2_DIR / '2-unified-data.pkl', 'wb') as f:
        pickle.dump(unified, f)
    print(f"  Saved: 2-unified-data.pkl")
    
    # Save contestant features as CSV
    features.to_csv(Q2_DIR / '2-contestant-features.csv', index=False)
    print(f"  Saved: 2-contestant-features.csv")
    
    # Save summary statistics
    summary = {
        'total_records': len(unified),
        'total_contestants': len(features),
        'seasons': sorted(unified['season'].unique().tolist()),
        'methods': unified['method'].unique().tolist(),
        'industries': sorted(features['industry_clean'].unique().tolist()),
        'dancers': sorted(features['ballroom_partner'].unique().tolist())
    }
    
    with open(Q2_DIR / '2-data-summary.pkl', 'wb') as f:
        pickle.dump(summary, f)
    print(f"  Saved: 2-data-summary.pkl")

def main():
    print("="*60)
    print("Question 2: Data Preparation")
    print("="*60)
    
    # Load data
    final_votes = load_question1_results()
    contestants = load_raw_data()
    
    # Create unified dataset
    unified = create_unified_dataset(final_votes, contestants)
    
    # Extract features
    features = extract_contestant_features(contestants)
    
    # Save outputs
    save_outputs(unified, features)
    
    print("\n" + "="*60)
    print("Data preparation complete!")
    print("="*60)
    print("\nOutput files:")
    print("  - Data/Question2/2-unified-data.pkl")
    print("  - Data/Question2/2-contestant-features.csv")
    print("  - Data/Question2/2-data-summary.pkl")

if __name__ == '__main__':
    main()
