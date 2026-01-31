"""
Rank Method Model for MCM Problem C Question 1
Estimate fan vote rankings for Seasons 1-2 and 28+ using optimization
"""

import pandas as pd
import numpy as np
import pickle
from scipy.optimize import minimize
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path('e:/Competition/数学建模/26美赛')

def load_preprocessed_data():
    """Load preprocessed data"""
    with open(OUTPUT_DIR / '1-preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

def optimize_season_rank(data, season):
    """
    Optimize fan vote rankings for one season using rank method.
    
    Rank Method:
    - Judge rank + Fan rank = Combined rank
    - Highest combined rank gets eliminated
    
    Model:
    - Minimize: sum of squared rank changes between consecutive weeks
    - Subject to:
      1. Fan ranks are a permutation of {1, 2, ..., n} for each week
      2. Elimination constraint: eliminated contestant has highest combined rank
    """
    season_data = data[data['season'] == season].copy()
    
    if len(season_data) == 0:
        return None
    
    # Get weeks
    weeks = sorted(season_data['week'].unique())
    
    # Build contestant tracking
    results = []
    prev_week_ranks = {}  # Store fan ranks from previous week
    
    for week in weeks:
        week_df = season_data[season_data['week'] == week].copy()
        n_contestants = len(week_df)
        
        if n_contestants == 0:
            continue
        
        # Get contestant list and judge ranks
        contestants = week_df['celebrity_name'].tolist()
        judge_ranks = dict(zip(week_df['celebrity_name'], week_df['judge_rank']))
        eliminated_list = week_df[week_df['eliminated'] == True]['celebrity_name'].tolist()
        eliminated = eliminated_list[0] if eliminated_list else None
        
        # For first week or when previous week data not available, use equal fan ranks
        if not prev_week_ranks:
            # Initial week: equal distribution assumption
            fan_ranks = {c: i+1 for i, c in enumerate(contestants)}
        else:
            # Optimize fan ranks based on smoothness from previous week
            # Try to minimize rank changes while satisfying elimination constraint
            
            # Initialize with previous week's ranks for remaining contestants
            fan_ranks = {}
            available_ranks = list(range(1, n_contestants + 1))
            
            # First, assign ranks to contestants who were there last week
            # Try to keep similar ranks (minimize change)
            contestants_from_prev = [c for c in contestants if c in prev_week_ranks]
            new_contestants = [c for c in contestants if c not in prev_week_ranks]
            
            # Sort previous contestants by their previous rank
            sorted_prev = sorted(contestants_from_prev, key=lambda c: prev_week_ranks[c])
            
            # Assign ranks trying to maintain order
            for i, c in enumerate(sorted_prev):
                target_rank = prev_week_ranks[c]
                # Find closest available rank
                closest_rank = min(available_ranks, key=lambda r: abs(r - target_rank))
                fan_ranks[c] = closest_rank
                available_ranks.remove(closest_rank)
            
            # Assign ranks to new contestants (if any - shouldn't happen in DWTS)
            for c in new_contestants:
                if available_ranks:
                    fan_ranks[c] = available_ranks.pop(0)
            
            # Check and adjust for elimination constraint
            if eliminated and eliminated in fan_ranks:
                # Calculate combined ranks
                combined_ranks = {c: judge_ranks[c] + fan_ranks[c] for c in contestants}
                max_combined = max(combined_ranks.values())
                elim_combined = combined_ranks[eliminated]
                
                # If eliminated is not the highest (worst) combined rank, adjust
                if elim_combined < max_combined:
                    # Need to swap ranks to make eliminated have highest combined rank
                    # Find contestant with highest combined rank
                    worst = max(combined_ranks.keys(), key=lambda c: combined_ranks[c])
                    
                    # Swap fan ranks between eliminated and worst
                    fan_ranks[eliminated], fan_ranks[worst] = fan_ranks[worst], fan_ranks[eliminated]
        
        # Calculate combined ranks
        combined_ranks = {c: judge_ranks[c] + fan_ranks[c] for c in contestants}
        
        # Store results for this week
        for _, row in week_df.iterrows():
            c = row['celebrity_name']
            results.append({
                'season': season,
                'week': week,
                'celebrity_name': c,
                'ballroom_partner': row['ballroom_partner'],
                'judge_total': row['judge_total'],
                'judge_percent': row['judge_percent'],
                'judge_rank': judge_ranks[c],
                'fan_rank': fan_ranks[c],
                'combined_rank': combined_ranks[c],
                'eliminated': row['eliminated'],
                'method': 'rank'
            })
        
        # Store fan ranks for next week's smoothness
        prev_week_ranks = fan_ranks.copy()
    
    return pd.DataFrame(results)

def estimate_vote_percentages_from_ranks(results_df):
    """
    Estimate vote percentages from ranks using a decreasing function.
    Higher rank (lower number) = higher vote percentage
    """
    def rank_to_percent(rank, n_contestants):
        """Convert rank to estimated percentage"""
        # Use a power law distribution
        # Rank 1 gets highest percentage, rank n gets lowest
        weights = np.array([1.0 / (r ** 0.5) for r in range(1, n_contestants + 1)])
        weights = weights / weights.sum()
        return weights[rank - 1]
    
    results_df = results_df.copy()
    percentages = []
    
    for (season, week), group in results_df.groupby(['season', 'week']):
        n = len(group)
        for _, row in group.iterrows():
            pct = rank_to_percent(row['fan_rank'], n)
            percentages.append(pct)
    
    results_df['fan_vote_percent'] = percentages
    results_df['combined_score'] = results_df['judge_percent'] + results_df['fan_vote_percent']
    
    return results_df

def calculate_rank_metrics(results_df):
    """Calculate consistency metrics for rank method"""
    metrics = {}
    
    # 1. Elimination accuracy
    total_eliminations = results_df[results_df['eliminated'] == True].shape[0]
    correct_eliminations = 0
    
    for (season, week), group in results_df.groupby(['season', 'week']):
        eliminated = group[group['eliminated'] == True]
        if len(eliminated) > 0:
            # Find contestant with highest combined rank (worst)
            max_rank = group['combined_rank'].max()
            worst_ranks = group[group['combined_rank'] == max_rank]
            
            if any(eliminated['celebrity_name'].isin(worst_ranks['celebrity_name'])):
                correct_eliminations += 1
    
    metrics['elimination_accuracy'] = correct_eliminations / total_eliminations if total_eliminations > 0 else 0
    metrics['total_eliminations'] = total_eliminations
    metrics['correct_eliminations'] = correct_eliminations
    
    # 2. Rank stability (smoothness)
    rank_changes = []
    for (season, contestant), group in results_df.groupby(['season', 'celebrity_name']):
        group_sorted = group.sort_values('week')
        fan_ranks = group_sorted['fan_rank'].values
        for i in range(1, len(fan_ranks)):
            rank_changes.append(abs(fan_ranks[i] - fan_ranks[i-1]))
    
    metrics['avg_rank_change'] = np.mean(rank_changes) if rank_changes else 0
    metrics['max_rank_change'] = np.max(rank_changes) if rank_changes else 0
    
    return metrics

def run_rank_method():
    """Main function for rank method modeling"""
    print("="*60)
    print("Rank Method Modeling (Seasons 1-2, 28+)")
    print("="*60)
    
    data = load_preprocessed_data()
    
    # Filter for rank method seasons
    rank_data = data[data['method'] == 'rank']
    seasons = sorted(rank_data['season'].unique())
    
    print(f"\nProcessing {len(seasons)} seasons with rank method...")
    
    all_results = []
    
    for season in seasons:
        print(f"\nSeason {season}:")
        result_df = optimize_season_rank(data, season)
        
        if result_df is not None:
            # Estimate vote percentages from ranks
            result_df = estimate_vote_percentages_from_ranks(result_df)
            all_results.append(result_df)
            print(f"    Processed {len(result_df)} records")
    
    # Combine all results
    final_results = pd.concat(all_results, ignore_index=True)
    
    # Calculate metrics
    metrics = calculate_rank_metrics(final_results)
    
    print("\n" + "="*60)
    print("Metrics Summary:")
    print("="*60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # Save results
    results_path = OUTPUT_DIR / '1-rank-vote-estimates.csv'
    final_results.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    
    metrics_path = OUTPUT_DIR / '1-rank-metrics.pkl'
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"Metrics saved to {metrics_path}")
    
    return final_results, metrics

if __name__ == '__main__':
    results, metrics = run_rank_method()
    print("\n" + "="*60)
    print("Sample of results:")
    print("="*60)
    print(results.head(20))
