"""
Percentage Method Model for MCM Problem C Question 1
Optimize to estimate fan vote percentages for Seasons 3-27
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

def optimize_season_percentage(data, season, alpha=0.5):
    """
    Optimize fan vote percentages for one season using percentage method.
    
    Model:
    - Minimize: sum of squared differences between consecutive weeks (smoothness)
    - Subject to:
      1. Sum of p_i,t = 1 for each week
      2. p_i,t >= 0
      3. Elimination constraint: eliminated contestant has lowest combined score
    """
    season_data = data[data['season'] == season].copy()
    
    if len(season_data) == 0:
        return None
    
    # Get weeks and contestants
    weeks = sorted(season_data['week'].unique())
    contestants = season_data['celebrity_name'].unique()
    
    # Build contestant-week mapping
    contestant_weeks = {}
    for contestant in contestants:
        contestant_data = season_data[season_data['celebrity_name'] == contestant]
        contestant_weeks[contestant] = {
            'weeks': sorted(contestant_data['week'].unique()),
            'data': contestant_data.set_index('week')
        }
    
    # Create optimization variables index
    # We'll have one variable per (contestant, week) pair
    var_list = []
    var_index = {}
    idx = 0
    
    for contestant in contestants:
        for week in contestant_weeks[contestant]['weeks']:
            var_list.append((contestant, week))
            var_index[(contestant, week)] = idx
            idx += 1
    
    n_vars = len(var_list)
    
    def objective(p):
        """Smoothness objective: minimize sum of squared week-to-week changes"""
        total = 0
        for contestant in contestants:
            weeks_list = contestant_weeks[contestant]['weeks']
            for i in range(1, len(weeks_list)):
                w_curr = weeks_list[i]
                w_prev = weeks_list[i-1]
                idx_curr = var_index[(contestant, w_curr)]
                idx_prev = var_index[(contestant, w_prev)]
                diff = p[idx_curr] - p[idx_prev]
                total += diff ** 2
        return total
    
    def constraint_sum_one(p, week_data):
        """Constraint: sum of percentages = 1 for each week"""
        week = week_data['week']
        contestants_in_week = week_data['contestants']
        total = sum(p[var_index[(c, week)]] for c in contestants_in_week if (c, week) in var_index)
        return total - 1.0
    
    def constraint_elimination(p, week_data, alpha):
        """Constraint: eliminated contestant has lowest combined score"""
        week = week_data['week']
        contestants_in_week = week_data['contestants']
        eliminated = week_data['eliminated']
        
        if not eliminated:
            return 0.0  # No elimination this week
        
        # Get eliminated contestant's combined score
        elim_contestant = eliminated[0]  # Assume one elimination per week
        if (elim_contestant, week) not in var_index:
            return 0.0
            
        idx_elim = var_index[(elim_contestant, week)]
        p_elim = p[idx_elim]
        q_elim = week_data['judge_percent'][elim_contestant]
        score_elim = alpha * q_elim + (1 - alpha) * p_elim
        
        # Constraint: score_elim <= score_all_others
        # We return the minimum margin (should be <= 0)
        margins = []
        for c in contestants_in_week:
            if c != elim_contestant and (c, week) in var_index:
                idx_c = var_index[(c, week)]
                p_c = p[idx_c]
                q_c = week_data['judge_percent'][c]
                score_c = alpha * q_c + (1 - alpha) * p_c
                margins.append(score_elim - score_c)
        
        if margins:
            return max(margins)  # Should be <= 0
        return 0.0
    
    # Prepare week data
    weeks_data = []
    for week in weeks:
        week_df = season_data[season_data['week'] == week]
        contestants_in_week = week_df['celebrity_name'].tolist()
        eliminated = week_df[week_df['eliminated'] == True]['celebrity_name'].tolist()
        judge_percent = dict(zip(week_df['celebrity_name'], week_df['judge_percent']))
        
        weeks_data.append({
            'week': week,
            'contestants': contestants_in_week,
            'eliminated': eliminated,
            'judge_percent': judge_percent
        })
    
    # Initial guess: equal distribution
    x0 = np.ones(n_vars) / 3  # Rough initial guess
    
    # Adjust initial guess based on judge percentages
    for i, (contestant, week) in enumerate(var_list):
        week_df = season_data[season_data['week'] == week]
        if len(week_df) > 0:
            x0[i] = 1.0 / len(week_df)
    
    # Build constraints
    constraints = []
    
    # Sum to 1 constraints
    for wd in weeks_data:
        constraints.append({
            'type': 'eq',
            'fun': lambda p, wd=wd: constraint_sum_one(p, wd)
        })
    
    # Elimination constraints (inequality: should be <= 0)
    for wd in weeks_data:
        if wd['eliminated']:
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, wd=wd: -constraint_elimination(p, wd, alpha)
            })
    
    # Bounds: 0 <= p <= 1
    bounds = [(0, 1) for _ in range(n_vars)]
    
    # Solve optimization problem
    print(f"    Solving optimization with {n_vars} variables and {len(constraints)} constraints...")
    
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-6}
    )
    
    if result.success:
        print(f"    Optimization successful! Final objective: {result.fun:.6f}")
    else:
        print(f"    Warning: Optimization did not converge. Status: {result.status}")
        print(f"    Message: {result.message}")
    
    # Build result dataframe
    results = []
    for i, (contestant, week) in enumerate(var_list):
        week_df = season_data[season_data['week'] == week]
        contestant_row = week_df[week_df['celebrity_name'] == contestant].iloc[0]
        
        results.append({
            'season': season,
            'week': week,
            'celebrity_name': contestant,
            'ballroom_partner': contestant_row['ballroom_partner'],
            'judge_total': contestant_row['judge_total'],
            'judge_percent': contestant_row['judge_percent'],
            'judge_rank': contestant_row['judge_rank'],
            'fan_vote_percent': result.x[i],
            'combined_score': alpha * contestant_row['judge_percent'] + (1 - alpha) * result.x[i],
            'eliminated': contestant_row['eliminated'],
            'method': 'percentage'
        })
    
    return pd.DataFrame(results)

def calculate_metrics(results_df):
    """Calculate consistency and certainty metrics"""
    metrics = {}
    
    # 1. Elimination accuracy
    total_eliminations = results_df[results_df['eliminated'] == True].shape[0]
    correct_eliminations = 0
    
    for (season, week), group in results_df.groupby(['season', 'week']):
        eliminated = group[group['eliminated'] == True]
        if len(eliminated) > 0:
            # Find contestant with lowest combined score
            min_score = group['combined_score'].min()
            lowest_scorers = group[group['combined_score'] == min_score]
            
            # Check if eliminated contestant is among lowest scorers
            if any(eliminated['celebrity_name'].isin(lowest_scorers['celebrity_name'])):
                correct_eliminations += 1
    
    metrics['elimination_accuracy'] = correct_eliminations / total_eliminations if total_eliminations > 0 else 0
    metrics['total_eliminations'] = total_eliminations
    metrics['correct_eliminations'] = correct_eliminations
    
    # 2. Elimination margin
    margins = []
    for (season, week), group in results_df.groupby(['season', 'week']):
        if group['eliminated'].any():
            sorted_scores = group['combined_score'].sort_values()
            if len(sorted_scores) >= 2:
                margin = sorted_scores.iloc[1] - sorted_scores.iloc[0]
                margins.append(margin)
    
    metrics['avg_elimination_margin'] = np.mean(margins) if margins else 0
    metrics['min_elimination_margin'] = np.min(margins) if margins else 0
    metrics['max_elimination_margin'] = np.max(margins) if margins else 0
    
    return metrics

def run_percentage_method():
    """Main function for percentage method modeling"""
    print("="*60)
    print("Percentage Method Modeling (Seasons 3-27)")
    print("="*60)
    
    data = load_preprocessed_data()
    
    # Filter for percentage method seasons
    percentage_data = data[data['method'] == 'percentage']
    seasons = sorted(percentage_data['season'].unique())
    
    print(f"\nProcessing {len(seasons)} seasons with percentage method...")
    
    all_results = []
    
    for season in seasons:
        print(f"\nSeason {season}:")
        result_df = optimize_season_percentage(data, season, alpha=0.5)
        
        if result_df is not None:
            all_results.append(result_df)
    
    # Combine all results
    final_results = pd.concat(all_results, ignore_index=True)
    
    # Calculate metrics
    metrics = calculate_metrics(final_results)
    
    print("\n" + "="*60)
    print("Metrics Summary:")
    print("="*60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # Save results
    results_path = OUTPUT_DIR / '1-percentage-vote-estimates.csv'
    final_results.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    
    metrics_path = OUTPUT_DIR / '1-percentage-metrics.pkl'
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"Metrics saved to {metrics_path}")
    
    return final_results, metrics

if __name__ == '__main__':
    results, metrics = run_percentage_method()
    print("\n" + "="*60)
    print("Sample of results:")
    print("="*60)
    print(results.head(20))
