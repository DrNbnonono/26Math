"""
Generate final combined results and metrics for Question 1
Creates the missing files:
- 1-final-vote-estimates.csv
- 1-consistency-metrics.json
- 1-uncertainty-bounds.csv
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

OUTPUT_DIR = Path('e:/Competition/数学建模/26美赛')

def generate_final_results():
    """Combine percentage and rank method results into final file"""
    # Load both results
    percentage_df = pd.read_csv(OUTPUT_DIR / '1-percentage-vote-estimates.csv')
    rank_df = pd.read_csv(OUTPUT_DIR / '1-rank-vote-estimates.csv')
    
    # Combine
    final_df = pd.concat([percentage_df, rank_df], ignore_index=True)
    
    # Sort by season and week
    final_df = final_df.sort_values(['season', 'week', 'celebrity_name']).reset_index(drop=True)
    
    # Save
    final_path = OUTPUT_DIR / '1-final-vote-estimates.csv'
    final_df.to_csv(final_path, index=False)
    print(f"Created: {final_path}")
    print(f"  Total records: {len(final_df)}")
    print(f"  Seasons: {sorted(final_df['season'].unique())}")
    
    return final_df

def generate_consistency_metrics(final_df):
    """Generate comprehensive consistency metrics"""
    metrics = {
        'overall': {},
        'percentage_method': {},
        'rank_method': {},
        'by_season': []
    }
    
    # Overall stats
    total_records = len(final_df)
    total_weeks = len(final_df.groupby(['season', 'week']))
    total_eliminations = final_df[final_df['eliminated'] == True].shape[0]
    
    metrics['overall'] = {
        'total_records': int(total_records),
        'total_weeks': int(total_weeks),
        'total_eliminations': int(total_eliminations),
        'seasons_covered': sorted(final_df['season'].unique().tolist()),
        'methods_used': final_df['method'].unique().tolist()
    }
    
    # Method-specific metrics
    for method in ['percentage', 'rank']:
        method_df = final_df[final_df['method'] == method]
        eliminations = method_df[method_df['eliminated'] == True]
        
        # Calculate accuracy
        correct = 0
        total = 0
        for (s, w), group in method_df.groupby(['season', 'week']):
            if group['eliminated'].any():
                total += 1
                elim_row = group[group['eliminated'] == True].iloc[0]
                
                if method == 'percentage':
                    min_score = group['combined_score'].min()
                    if abs(elim_row['combined_score'] - min_score) < 1e-6:
                        correct += 1
                else:  # rank
                    max_rank = group['combined_rank'].max()
                    if abs(elim_row['combined_rank'] - max_rank) < 1e-6:
                        correct += 1
        
        accuracy = correct / total if total > 0 else 0
        
        metrics[f'{method}_method'] = {
            'total_records': int(len(method_df)),
            'total_eliminations': int(len(eliminations)),
            'prediction_accuracy': float(accuracy),
            'correct_predictions': int(correct),
            'total_prediction_weeks': int(total)
        }
    
    # Per-season metrics
    for season in sorted(final_df['season'].unique()):
        season_df = final_df[final_df['season'] == season]
        method = season_df.iloc[0]['method']
        
        eliminations = season_df[season_df['eliminated'] == True]
        
        # Calculate accuracy
        correct = 0
        total = 0
        for w, group in season_df.groupby('week'):
            if group['eliminated'].any():
                total += 1
                elim_row = group[group['eliminated'] == True].iloc[0]
                
                if method == 'percentage':
                    min_score = group['combined_score'].min()
                    if abs(elim_row['combined_score'] - min_score) < 1e-6:
                        correct += 1
                else:
                    max_rank = group['combined_rank'].max()
                    if abs(elim_row['combined_rank'] - max_rank) < 1e-6:
                        correct += 1
        
        accuracy = correct / total if total > 0 else 0
        
        metrics['by_season'].append({
            'season': int(season),
            'method': method,
            'contestants': int(season_df['celebrity_name'].nunique()),
            'weeks': int(season_df['week'].nunique()),
            'eliminations': int(len(eliminations)),
            'prediction_accuracy': float(accuracy),
            'avg_fan_vote_percent': float(season_df['fan_vote_percent'].mean()),
            'avg_judge_percent': float(season_df['judge_percent'].mean())
        })
    
    # Save
    metrics_path = OUTPUT_DIR / '1-consistency-metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Created: {metrics_path}")
    
    return metrics

def generate_uncertainty_bounds(final_df):
    """Generate uncertainty bounds for each contestant-week"""
    bounds_list = []
    
    for (season, contestant), group in final_df.groupby(['season', 'celebrity_name']):
        group_sorted = group.sort_values('week')
        
        # Calculate statistics across weeks
        vote_values = group_sorted['fan_vote_percent'].values
        
        for _, row in group_sorted.iterrows():
            # Current week's vote
            current_vote = row['fan_vote_percent']
            
            # Calculate bounds based on overall variation
            # This is a simplified approach - in a full implementation,
            # you would solve optimization problems to find true bounds
            vote_std = vote_values.std() if len(vote_values) > 1 else 0
            vote_mean = vote_values.mean()
            
            # Estimate bounds as mean ± 2*std (simplified)
            lower_bound = max(0, current_vote - 2 * vote_std)
            upper_bound = min(1, current_vote + 2 * vote_std)
            
            # Uncertainty measure
            uncertainty = upper_bound - lower_bound
            
            # Relative uncertainty (coefficient of variation style)
            relative_uncertainty = uncertainty / current_vote if current_vote > 0 else 0
            
            bounds_list.append({
                'season': int(row['season']),
                'week': int(row['week']),
                'celebrity_name': row['celebrity_name'],
                'ballroom_partner': row['ballroom_partner'],
                'method': row['method'],
                'estimated_vote_percent': float(current_vote),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'uncertainty_range': float(uncertainty),
                'relative_uncertainty': float(relative_uncertainty),
                'judge_percent': float(row['judge_percent']),
                'combined_score': float(row['combined_score']) if 'combined_score' in row else None,
                'eliminated': bool(row['eliminated'])
            })
    
    bounds_df = pd.DataFrame(bounds_list)
    
    # Save
    bounds_path = OUTPUT_DIR / '1-uncertainty-bounds.csv'
    bounds_df.to_csv(bounds_path, index=False)
    print(f"Created: {bounds_path}")
    print(f"  Total records: {len(bounds_df)}")
    print(f"  Average uncertainty: {bounds_df['uncertainty_range'].mean():.4f}")
    
    return bounds_df

if __name__ == '__main__':
    print("="*60)
    print("Generating Final Result Files")
    print("="*60)
    
    # Generate all missing files
    final_df = generate_final_results()
    print()
    
    metrics = generate_consistency_metrics(final_df)
    print()
    
    bounds_df = generate_uncertainty_bounds(final_df)
    print()
    
    print("="*60)
    print("All files generated successfully!")
    print("="*60)
    
    # Summary
    print("\nFile Summary:")
    print(f"1. 1-final-vote-estimates.csv: {len(final_df)} records")
    print(f"2. 1-consistency-metrics.json: Metrics for {len(metrics['by_season'])} seasons")
    print(f"3. 1-uncertainty-bounds.csv: {len(bounds_df)} records with uncertainty bounds")
