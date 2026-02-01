"""
Question 3: Data Preparation
Build feature matrix for regression analysis
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path('e:/Competition/数学建模/26美赛')
DATA_DIR = BASE_DIR / 'Data'
Q1_DIR = DATA_DIR / 'Question1'
Q3_DIR = DATA_DIR / 'Question3'
Q3_DIR.mkdir(exist_ok=True)

def load_data():
    """Load Q1 results and raw data"""
    print("Loading data...")
    
    # Load Q1 fan vote estimates
    final_votes = pd.read_csv(Q1_DIR / '1-final-vote-estimates.csv')
    
    # Load raw data
    raw_data = pd.read_csv(DATA_DIR / '2026_MCM_Problem_C_Data.csv')
    
    print(f"  Q1 results: {len(final_votes)} records")
    print(f"  Raw data: {len(raw_data)} contestants")
    
    return final_votes, raw_data

def build_feature_matrix(final_votes, raw_data):
    """Build comprehensive feature matrix"""
    print("\nBuilding feature matrix...")
    
    # Merge data
    features = final_votes.copy()
    
    # Add contestant characteristics from raw data
    contestant_info = raw_data[['celebrity_name', 'season', 'celebrity_industry', 
                                'celebrity_homestate', 'celebrity_homecountry/region',
                                'celebrity_age_during_season', 'placement']].copy()
    contestant_info.columns = ['celebrity_name', 'season', 'industry', 'home_state', 
                               'home_country', 'age', 'final_placement']
    contestant_info = contestant_info.drop_duplicates(subset=['season', 'celebrity_name'])
    
    features = features.merge(contestant_info, on=['season', 'celebrity_name'], how='left')
    
    # Calculate judge average score (normalized for 3 vs 4 judges)
    # Count number of judges per week from raw data
    judge_cols = [col for col in raw_data.columns if 'judge' in col and 'score' in col]
    
    # For each contestant-week, calculate average judge score
    features['judge_avg_score'] = features['judge_total'] / 3  # Approximate, will refine
    
    # Calculate survival weeks
    survival = features.groupby(['season', 'celebrity_name'])['week'].max().reset_index()
    survival.columns = ['season', 'celebrity_name', 'survival_weeks']
    features = features.merge(survival, on=['season', 'celebrity_name'], how='left')
    
    # Clean categorical variables
    features['industry_clean'] = features['industry'].fillna('Unknown')
    features['home_state_clean'] = features['home_state'].fillna('Unknown')
    features['home_country_clean'] = features['home_country'].fillna('Unknown')
    
    # Create age groups
    features['age_group'] = pd.cut(features['age'], bins=[0, 30, 40, 50, 100],
                                   labels=['<30', '30-40', '40-50', '50+'])
    
    print(f"  Feature matrix: {len(features)} records")
    print(f"  Features: {features.columns.tolist()}")
    
    return features

def check_multicollinearity(features):
    """Check for multicollinearity using VIF"""
    print("\nChecking multicollinearity...")
    
    # Select numeric features for VIF calculation
    numeric_features = features[['judge_total', 'judge_percent', 'fan_vote_percent', 
                                 'age', 'week']].dropna()
    
    # Calculate VIF
    vif_data = pd.DataFrame()
    vif_data['Feature'] = numeric_features.columns
    vif_data['VIF'] = [variance_inflation_factor(numeric_features.values, i) 
                       for i in range(len(numeric_features.columns))]
    
    print("\n  VIF Scores:")
    print(vif_data.to_string(index=False))
    
    # Flag high VIF (>5 or >10)
    high_vif = vif_data[vif_data['VIF'] > 5]
    if len(high_vif) > 0:
        print(f"\n  WARNING: {len(high_vif)} features with VIF > 5 (potential multicollinearity)")
        print(high_vif.to_string(index=False))
    else:
        print("\n  No severe multicollinearity detected (all VIF < 5)")
    
    # Save VIF report
    with open(Q3_DIR / '3-multicollinearity-report.txt', 'w') as f:
        f.write("Variance Inflation Factor (VIF) Analysis\n")
        f.write("="*50 + "\n\n")
        f.write(vif_data.to_string(index=False))
        f.write("\n\n")
        if len(high_vif) > 0:
            f.write("High VIF Features (>5):\n")
            f.write(high_vif.to_string(index=False))
        else:
            f.write("No severe multicollinearity detected.")
    
    print("  Saved: 3-multicollinearity-report.txt")
    
    return vif_data

def create_modeling_datasets(features):
    """Create separate datasets for different modeling tasks"""
    print("\nCreating modeling datasets...")
    
    # Dataset A: For judge score prediction (weekly level)
    judge_data = features[['season', 'week', 'celebrity_name', 'ballroom_partner',
                          'judge_avg_score', 'judge_total', 'age', 'industry_clean',
                          'home_state_clean', 'home_country_clean']].copy()
    
    # Dataset B: For fan vote prediction (weekly level)
    fan_data = features[['season', 'week', 'celebrity_name', 'ballroom_partner',
                        'fan_vote_percent', 'age', 'industry_clean',
                        'home_state_clean', 'home_country_clean']].copy()
    
    # Dataset C: For final placement prediction (contestant level)
    placement_data = features.groupby(['season', 'celebrity_name']).agg({
        'ballroom_partner': 'first',
        'age': 'first',
        'industry_clean': 'first',
        'home_state_clean': 'first',
        'home_country_clean': 'first',
        'survival_weeks': 'first',
        'final_placement': 'first',
        'judge_avg_score': 'mean',
        'fan_vote_percent': 'mean'
    }).reset_index()
    
    print(f"  Judge score dataset: {len(judge_data)} records")
    print(f"  Fan vote dataset: {len(fan_data)} records")
    print(f"  Placement dataset: {len(placement_data)} records")
    
    return judge_data, fan_data, placement_data

def save_outputs(features, judge_data, fan_data, placement_data, vif_data):
    """Save all preprocessed data"""
    print("\nSaving outputs...")
    
    # Save full feature matrix
    with open(Q3_DIR / '3-features.pkl', 'wb') as f:
        pickle.dump(features, f)
    print("  Saved: 3-features.pkl")
    
    # Save modeling datasets
    judge_data.to_csv(Q3_DIR / '3-judge-data.csv', index=False)
    print("  Saved: 3-judge-data.csv")
    
    fan_data.to_csv(Q3_DIR / '3-fan-data.csv', index=False)
    print("  Saved: 3-fan-data.csv")
    
    placement_data.to_csv(Q3_DIR / '3-placement-data.csv', index=False)
    print("  Saved: 3-placement-data.csv")
    
    # Save summary statistics
    summary = {
        'total_records': len(features),
        'total_contestants': features.groupby(['season', 'celebrity_name']).ngroups,
        'total_seasons': features['season'].nunique(),
        'industries': sorted(features['industry_clean'].unique().tolist()),
        'dancers': sorted(features['ballroom_partner'].unique().tolist()),
        'vif_summary': vif_data.to_dict('records')
    }
    
    with open(Q3_DIR / '3-data-summary.pkl', 'wb') as f:
        pickle.dump(summary, f)
    print("  Saved: 3-data-summary.pkl")

def main():
    print("="*60)
    print("Question 3: Data Preparation")
    print("="*60)
    
    # Load data
    final_votes, raw_data = load_data()
    
    # Build feature matrix
    features = build_feature_matrix(final_votes, raw_data)
    
    # Check multicollinearity
    vif_data = check_multicollinearity(features)
    
    # Create modeling datasets
    judge_data, fan_data, placement_data = create_modeling_datasets(features)
    
    # Save outputs
    save_outputs(features, judge_data, fan_data, placement_data, vif_data)
    
    print("\n" + "="*60)
    print("Data preparation complete!")
    print("="*60)

if __name__ == '__main__':
    main()
