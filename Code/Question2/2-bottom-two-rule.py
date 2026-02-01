"""
Question 2: Bottom Two Rule Simulation
Simulate the "judges choose from bottom 2" elimination rule
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

BASE_DIR = Path('e:/Competition/数学建模/26美赛')
Q2_DIR = BASE_DIR / 'Data' / 'Question2'

def load_data():
    with open(Q2_DIR / '2-unified-data.pkl', 'rb') as f:
        return pickle.load(f)

def simulate_bottom_two_rule(data, season):
    """
    Simulate: judges choose which of bottom 2 to eliminate
    Assumption: judges choose the one with lower judge score
    """
    season_data = data[data['season'] == season].copy()
    weeks = sorted(season_data['week'].unique())
    
    active_contestants = set(season_data['celebrity_name'].unique())
    eliminations = []
    
    for week in weeks:
        week_data = season_data[season_data['week'] == week].copy()
        week_active = week_data[week_data['celebrity_name'].isin(active_contestants)]
        
        if len(week_active) <= 2:
            continue
        
        # Calculate combined scores (percentage method)
        judge_sum = week_active['judge_total'].sum()
        fan_sum = week_active['fan_vote_percent'].sum()
        week_active['judge_pct'] = week_active['judge_total'] / judge_sum
        week_active['fan_pct'] = week_active['fan_vote_percent'] / fan_sum
        week_active['combined'] = week_active['judge_pct'] + week_active['fan_pct']
        
        # Find bottom 2
        bottom_2 = week_active.nsmallest(2, 'combined')
        
        # Judges choose: eliminate the one with lower judge score
        eliminated = bottom_2.loc[bottom_2['judge_total'].idxmin(), 'celebrity_name']
        
        eliminations.append({
            'week': int(week),
            'eliminated': eliminated,
            'bottom_2': bottom_2['celebrity_name'].tolist()
        })
        
        active_contestants.discard(eliminated)
    
    return {
        'season': int(season),
        'eliminations': eliminations,
        'final_survivors': list(active_contestants)
    }

def main():
    print("="*60)
    print("Question 2: Bottom Two Rule Simulation")
    print("="*60)
    
    data = load_data()
    seasons = sorted(data['season'].unique())
    
    all_results = []
    for season in seasons:
        result = simulate_bottom_two_rule(data, season)
        all_results.append(result)
    
    # Save results
    results_df = pd.DataFrame([{
        'season': r['season'],
        'num_eliminations': len(r['eliminations'])
    } for r in all_results])
    
    results_df.to_csv(Q2_DIR / '2-bottom-two-results.csv', index=False)
    print("  Saved: 2-bottom-two-results.csv")
    
    print("\n" + "="*60)
    print("Bottom two rule simulation complete!")
    print("="*60)

if __name__ == '__main__':
    main()
