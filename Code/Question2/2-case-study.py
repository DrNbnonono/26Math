"""
Question 2: Case Study Analysis
Analyze the 4 specified controversial contestants
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

BASE_DIR = Path('e:/Competition/数学建模/26美赛')
Q2_DIR = BASE_DIR / 'Data' / 'Question2'

def load_data():
    with open(Q2_DIR / '2-unified-data.pkl', 'rb') as f:
        return pickle.load(f)

def analyze_case(data, name, season):
    """Analyze one controversial case"""
    contestant_data = data[(data['celebrity_name'] == name) & (data['season'] == season)].copy()
    
    if len(contestant_data) == 0:
        return None
    
    contestant_data = contestant_data.sort_values('week')
    
    # For each week, check if they would be eliminated under the OTHER method
    results = []
    for _, row in contestant_data.iterrows():
        week = row['week']
        week_data = data[(data['season'] == season) & (data['week'] == week)]
        
        # Current method (from data)
        current_method = row['method']
        
        # Simulate the OTHER method
        if current_method == 'percentage':
            # Check if would be eliminated under rank method
            week_data = week_data.copy()
            week_data['judge_rank_sim'] = week_data['judge_total'].rank(ascending=False, method='min')
            week_data['fan_rank_sim'] = week_data['fan_vote_percent'].rank(ascending=False, method='min')
            week_data['combined_rank_sim'] = week_data['judge_rank_sim'] + week_data['fan_rank_sim']
            
            max_rank = week_data['combined_rank_sim'].max()
            contestant_rank = week_data[week_data['celebrity_name'] == name]['combined_rank_sim'].iloc[0]
            would_be_eliminated = (contestant_rank == max_rank)
        else:
            # Check if would be eliminated under percentage method
            judge_sum = week_data['judge_total'].sum()
            fan_sum = week_data['fan_vote_percent'].sum()
            week_data = week_data.copy()
            week_data['judge_pct_sim'] = week_data['judge_total'] / judge_sum
            week_data['fan_pct_sim'] = week_data['fan_vote_percent'] / fan_sum
            week_data['combined_pct_sim'] = week_data['judge_pct_sim'] + week_data['fan_pct_sim']
            
            min_score = week_data['combined_pct_sim'].min()
            contestant_score = week_data[week_data['celebrity_name'] == name]['combined_pct_sim'].iloc[0]
            would_be_eliminated = (contestant_score == min_score)
        
        results.append({
            'week': int(week),
            'judge_rank': int(row['judge_rank']),
            'fan_vote_percent': float(row['fan_vote_percent']),
            'current_method': current_method,
            'would_be_eliminated_other_method': bool(would_be_eliminated),
            'actually_eliminated': bool(row['eliminated'])
        })
    
    return {
        'name': name,
        'season': int(season),
        'final_placement': int(contestant_data['final_placement'].iloc[0]) if pd.notna(contestant_data['final_placement'].iloc[0]) else None,
        'weeks_survived': len(results),
        'weekly_analysis': results
    }

def main():
    print("="*60)
    print("Question 2: Case Study Analysis")
    print("="*60)
    
    data = load_data()
    
    cases = [
        ('Jerry Rice', 2),
        ('Billy Ray Cyrus', 4),
        ('Bristol Palin', 11),
        ('Bobby Bones', 27)
    ]
    
    case_results = []
    for name, season in cases:
        print(f"\nAnalyzing {name} (Season {season})...")
        result = analyze_case(data, name, season)
        if result:
            case_results.append(result)
            weeks_would_be_elim = sum(1 for w in result['weekly_analysis'] if w['would_be_eliminated_other_method'])
            print(f"  Survived {result['weeks_survived']} weeks, final placement: {result['final_placement']}")
            print(f"  Would have been eliminated {weeks_would_be_elim} times under other method")
        else:
            print(f"  NOT FOUND in data")
    
    # Save results
    with open(Q2_DIR / '2-case-study-results.json', 'w') as f:
        json.dump(case_results, f, indent=2)
    print("\n  Saved: 2-case-study-results.json")
    
    print("\n" + "="*60)
    print("Case study analysis complete!")
    print("="*60)

if __name__ == '__main__':
    main()
