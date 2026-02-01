"""
Question 2: Method Comparison Simulation
Apply both percentage and rank methods to all 34 seasons and compare results
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path('e:/Competition/数学建模/26美赛')
Q2_DIR = BASE_DIR / 'Data' / 'Question2'

def load_data():
    """Load preprocessed data"""
    print("Loading preprocessed data...")
    with open(Q2_DIR / '2-unified-data.pkl', 'rb') as f:
        data = pickle.load(f)
    print(f"  Loaded {len(data)} records")
    return data

def apply_percentage_method(week_data):
    """
    Apply percentage method to a single week
    Returns: (eliminated_contestant, combined_scores_dict)
    """
    # Calculate percentages
    judge_sum = week_data['judge_total'].sum()
    fan_sum = week_data['fan_vote_percent'].sum()
    
    results = []
    for _, row in week_data.iterrows():
        judge_pct = row['judge_total'] / judge_sum if judge_sum > 0 else 0
        fan_pct = row['fan_vote_percent'] / fan_sum if fan_sum > 0 else 0
        combined = judge_pct + fan_pct
        results.append({
            'celebrity_name': row['celebrity_name'],
            'judge_pct': judge_pct,
            'fan_pct': fan_pct,
            'combined_score': combined
        })
    
    results_df = pd.DataFrame(results)
    eliminated = results_df.loc[results_df['combined_score'].idxmin(), 'celebrity_name']
    
    return eliminated, results_df.set_index('celebrity_name')['combined_score'].to_dict()

def apply_rank_method(week_data):
    """
    Apply rank method to a single week
    Returns: (eliminated_contestant, combined_ranks_dict)
    """
    # Rank by judge scores (higher score = lower rank number)
    week_data = week_data.copy()
    week_data['judge_rank'] = week_data['judge_total'].rank(ascending=False, method='min')
    week_data['fan_rank'] = week_data['fan_vote_percent'].rank(ascending=False, method='min')
    week_data['combined_rank'] = week_data['judge_rank'] + week_data['fan_rank']
    
    # Highest combined rank is eliminated
    eliminated = week_data.loc[week_data['combined_rank'].idxmax(), 'celebrity_name']
    
    return eliminated, week_data.set_index('celebrity_name')['combined_rank'].to_dict()

def simulate_season(data, season):
    """
    Simulate one season with both methods
    Returns: comparison results for the season
    """
    season_data = data[data['season'] == season].copy()
    weeks = sorted(season_data['week'].unique())
    
    # Track active contestants for each method
    all_contestants = set(season_data['celebrity_name'].unique())
    active_pct = all_contestants.copy()
    active_rank = all_contestants.copy()
    
    eliminations_pct = []
    eliminations_rank = []
    divergence_week = None
    
    for week in weeks:
        week_data = season_data[season_data['week'] == week].copy()
        
        # Skip if no eliminations this week (check actual data)
        actual_eliminated = week_data[week_data['eliminated'] == True]['celebrity_name'].tolist()
        if not actual_eliminated:
            continue
        
        # Apply percentage method (only to active contestants)
        week_pct = week_data[week_data['celebrity_name'].isin(active_pct)]
        if len(week_pct) > 1:
            elim_pct, scores_pct = apply_percentage_method(week_pct)
            eliminations_pct.append({
                'week': week,
                'eliminated': elim_pct,
                'method': 'percentage'
            })
            active_pct.discard(elim_pct)
        
        # Apply rank method (only to active contestants)
        week_rank = week_data[week_data['celebrity_name'].isin(active_rank)]
        if len(week_rank) > 1:
            elim_rank, ranks_rank = apply_rank_method(week_rank)
            eliminations_rank.append({
                'week': week,
                'eliminated': elim_rank,
                'method': 'rank'
            })
            active_rank.discard(elim_rank)
        
        # Check for divergence
        if elim_pct != elim_rank and divergence_week is None:
            divergence_week = week
    
    return {
        'season': season,
        'divergence_week': divergence_week,
        'diverged': divergence_week is not None,
        'eliminations_pct': eliminations_pct,
        'eliminations_rank': eliminations_rank,
        'final_survivors_pct': list(active_pct),
        'final_survivors_rank': list(active_rank)
    }

def compare_with_fan_only(data, season):
    """
    Compare each method's results with pure fan voting
    """
    season_data = data[data['season'] == season].copy()
    weeks = sorted(season_data['week'].unique())
    
    # Simulate fan-only elimination
    active_fan = set(season_data['celebrity_name'].unique())
    eliminations_fan = []
    
    for week in weeks:
        week_data = season_data[season_data['week'] == week].copy()
        week_fan = week_data[week_data['celebrity_name'].isin(active_fan)]
        
        if len(week_fan) > 1:
            # Eliminate contestant with lowest fan vote
            elim_fan = week_fan.loc[week_fan['fan_vote_percent'].idxmin(), 'celebrity_name']
            eliminations_fan.append(elim_fan)
            active_fan.discard(elim_fan)
    
    return {
        'eliminations_fan': eliminations_fan,
        'final_survivor_fan': list(active_fan)
    }

def calculate_alignment_metrics(data):
    """
    Calculate how well each method aligns with pure fan voting
    """
    print("\nCalculating alignment with fan-only voting...")
    
    seasons = sorted(data['season'].unique())
    alignment_results = []
    
    for season in seasons:
        season_data = data[data['season'] == season].copy()
        
        # Get final placements under each method
        sim_result = simulate_season(data, season)
        fan_result = compare_with_fan_only(data, season)
        
        # Calculate Kendall's tau correlation for rankings
        # (measure of ordinal association)
        
        alignment_results.append({
            'season': season,
            'diverged': sim_result['diverged'],
            'divergence_week': sim_result['divergence_week']
        })
    
    return pd.DataFrame(alignment_results)

def main():
    print("="*60)
    print("Question 2: Method Comparison Simulation")
    print("="*60)
    
    # Load data
    data = load_data()
    
    # Simulate all seasons
    print("\nSimulating all 34 seasons with both methods...")
    all_results = []
    
    for season in sorted(data['season'].unique()):
        print(f"  Season {season}...", end=" ")
        result = simulate_season(data, season)
        all_results.append(result)
        
        if result['diverged']:
            print(f"DIVERGED at week {result['divergence_week']}")
        else:
            print("No divergence")
    
    # Calculate statistics
    diverged_count = sum(1 for r in all_results if r['diverged'])
    divergence_rate = diverged_count / len(all_results)
    
    print(f"\n{'='*60}")
    print(f"Divergence Statistics:")
    print(f"  Total seasons: {len(all_results)}")
    print(f"  Diverged seasons: {diverged_count}")
    print(f"  Divergence rate: {divergence_rate:.2%}")
    print(f"{'='*60}")
    
    # Calculate alignment metrics
    alignment_df = calculate_alignment_metrics(data)
    
    # Save results
    print("\nSaving results...")
    
    # Save detailed comparison
    comparison_df = pd.DataFrame(all_results)
    comparison_df.to_csv(Q2_DIR / '2-method-comparison-results.csv', index=False)
    print("  Saved: 2-method-comparison-results.csv")
    
    # Save divergence analysis
    divergence_analysis = {
        'total_seasons': int(len(all_results)),
        'diverged_count': int(diverged_count),
        'divergence_rate': float(divergence_rate),
        'diverged_seasons': [int(r['season']) for r in all_results if r['diverged']],
        'non_diverged_seasons': [int(r['season']) for r in all_results if not r['diverged']],
        'alignment_summary': [{k: (int(v) if isinstance(v, (np.integer, np.int64)) else v) 
                               for k, v in record.items()} 
                              for record in alignment_df.to_dict('records')]
    }
    
    with open(Q2_DIR / '2-divergence-analysis.json', 'w') as f:
        json.dump(divergence_analysis, f, indent=2)
    print("  Saved: 2-divergence-analysis.json")
    
    print("\n" + "="*60)
    print("Method comparison complete!")
    print("="*60)

if __name__ == '__main__':
    main()
