"""
Question 2: Method Bias Analysis
Analyze which method (percentage vs rank) is more biased toward fan votes
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from scipy import stats

BASE_DIR = Path('e:/Competition/数学建模/26美赛')
Q2_DIR = BASE_DIR / 'Data' / 'Question2'

def load_data():
    """Load preprocessed data"""
    with open(Q2_DIR / '2-unified-data.pkl', 'rb') as f:
        return pickle.load(f)

def calculate_method_bias(data):
    """
    Calculate bias toward fan votes for each method
    Bias metric: correlation between elimination decision and fan vote rank
    Higher correlation = more bias toward fans
    """
    print("\nCalculating method bias toward fan votes...")
    
    results = {
        'percentage_method': [],
        'rank_method': []
    }
    
    # Analyze each season
    for season in sorted(data['season'].unique()):
        season_data = data[data['season'] == season].copy()
        method = season_data['method'].iloc[0]
        
        # For each elimination week
        elimination_weeks = season_data[season_data['eliminated'] == True]['week'].unique()
        
        for week in elimination_weeks:
            week_data = season_data[season_data['week'] == week].copy()
            
            if len(week_data) < 2:
                continue
            
            # Calculate fan vote rank (lower is better)
            week_data['fan_rank'] = week_data['fan_vote_percent'].rank(ascending=False, method='min')
            
            # Calculate judge rank
            week_data['judge_rank_calc'] = week_data['judge_total'].rank(ascending=False, method='min')
            
            # Get eliminated contestant
            eliminated = week_data[week_data['eliminated'] == True]
            if len(eliminated) == 0:
                continue
            
            eliminated_name = eliminated['celebrity_name'].iloc[0]
            eliminated_fan_rank = eliminated['fan_rank'].iloc[0]
            eliminated_judge_rank = eliminated['judge_rank_calc'].iloc[0]
            
            # Calculate how much the elimination aligns with fan votes vs judge scores
            # If eliminated contestant has worst fan rank, bias toward fans = 1
            # If eliminated contestant has best fan rank, bias toward fans = 0
            max_fan_rank = week_data['fan_rank'].max()
            fan_alignment = eliminated_fan_rank / max_fan_rank  # 0-1, higher = more aligned with fans
            
            max_judge_rank = week_data['judge_rank_calc'].max()
            judge_alignment = eliminated_judge_rank / max_judge_rank
            
            # Bias score: positive = more aligned with fans, negative = more aligned with judges
            bias_score = fan_alignment - judge_alignment
            
            if method == 'percentage':
                results['percentage_method'].append({
                    'season': season,
                    'week': week,
                    'fan_alignment': fan_alignment,
                    'judge_alignment': judge_alignment,
                    'bias_score': bias_score
                })
            else:
                results['rank_method'].append({
                    'season': season,
                    'week': week,
                    'fan_alignment': fan_alignment,
                    'judge_alignment': judge_alignment,
                    'bias_score': bias_score
                })
    
    return results

def analyze_fan_vote_weight(data):
    """
    Analyze the effective weight of fan votes in each method
    by comparing elimination outcomes with pure fan voting
    """
    print("\nAnalyzing effective fan vote weight...")
    
    method_weights = {
        'percentage': [],
        'rank': []
    }
    
    for season in sorted(data['season'].unique()):
        season_data = data[data['season'] == season].copy()
        method = season_data['method'].iloc[0]
        
        # Count how many times the eliminated contestant had the lowest fan vote
        elimination_weeks = season_data[season_data['eliminated'] == True]['week'].unique()
        
        fan_vote_eliminations = 0
        total_eliminations = 0
        
        for week in elimination_weeks:
            week_data = season_data[season_data['week'] == week].copy()
            
            if len(week_data) < 2:
                continue
            
            # Get eliminated contestant
            eliminated = week_data[week_data['eliminated'] == True]
            if len(eliminated) == 0:
                continue
            
            eliminated_name = eliminated['celebrity_name'].iloc[0]
            
            # Check if this contestant had the lowest fan vote
            min_fan_vote = week_data['fan_vote_percent'].min()
            eliminated_fan_vote = eliminated['fan_vote_percent'].iloc[0]
            
            if abs(eliminated_fan_vote - min_fan_vote) < 0.001:  # Accounting for floating point
                fan_vote_eliminations += 1
            
            total_eliminations += 1
        
        if total_eliminations > 0:
            fan_vote_weight = fan_vote_eliminations / total_eliminations
            
            if method == 'percentage':
                method_weights['percentage'].append(fan_vote_weight)
            else:
                method_weights['rank'].append(fan_vote_weight)
    
    return method_weights

def statistical_test(bias_results):
    """
    Perform statistical test to determine if one method is significantly
    more biased toward fan votes
    """
    print("\nPerforming statistical test...")
    
    pct_bias = [r['bias_score'] for r in bias_results['percentage_method']]
    rank_bias = [r['bias_score'] for r in bias_results['rank_method']]
    
    # T-test
    t_stat, p_value = stats.ttest_ind(pct_bias, rank_bias)
    
    # Calculate means
    pct_mean = np.mean(pct_bias)
    rank_mean = np.mean(rank_bias)
    
    print(f"\n  Percentage Method:")
    print(f"    Mean bias score: {pct_mean:.4f}")
    print(f"    Std deviation: {np.std(pct_bias):.4f}")
    print(f"    Sample size: {len(pct_bias)}")
    
    print(f"\n  Rank Method:")
    print(f"    Mean bias score: {rank_mean:.4f}")
    print(f"    Std deviation: {np.std(rank_bias):.4f}")
    print(f"    Sample size: {len(rank_bias)}")
    
    print(f"\n  T-test Results:")
    print(f"    t-statistic: {t_stat:.4f}")
    print(f"    p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        if pct_mean > rank_mean:
            conclusion = "Percentage method is SIGNIFICANTLY more biased toward fan votes"
        else:
            conclusion = "Rank method is SIGNIFICANTLY more biased toward fan votes"
    else:
        conclusion = "No significant difference in fan vote bias between methods"
    
    print(f"\n  Conclusion: {conclusion}")
    
    return {
        'percentage_mean': float(pct_mean),
        'rank_mean': float(rank_mean),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'conclusion': conclusion
    }

def main():
    print("="*60)
    print("Question 2: Method Bias Analysis")
    print("="*60)
    
    # Load data
    data = load_data()
    
    # Calculate bias scores
    bias_results = calculate_method_bias(data)
    
    # Analyze fan vote weight
    fan_weights = analyze_fan_vote_weight(data)
    
    print(f"\n{'='*60}")
    print("Fan Vote Alignment Analysis")
    print(f"{'='*60}")
    
    pct_weight = np.mean(fan_weights['percentage'])
    rank_weight = np.mean(fan_weights['rank'])
    
    print(f"\nPercentage Method:")
    print(f"  Average alignment with pure fan voting: {pct_weight:.2%}")
    print(f"  ({pct_weight*100:.1f}% of eliminations matched lowest fan vote)")
    
    print(f"\nRank Method:")
    print(f"  Average alignment with pure fan voting: {rank_weight:.2%}")
    print(f"  ({rank_weight*100:.1f}% of eliminations matched lowest fan vote)")
    
    # Statistical test
    test_results = statistical_test(bias_results)
    
    # Save results
    print("\nSaving results...")
    
    # Convert all values to JSON-serializable types
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    analysis_results = {
        'bias_scores': {
            'percentage_method': convert_to_serializable(bias_results['percentage_method']),
            'rank_method': convert_to_serializable(bias_results['rank_method'])
        },
        'fan_vote_alignment': {
            'percentage_method': float(pct_weight),
            'rank_method': float(rank_weight)
        },
        'statistical_test': test_results
    }
    
    with open(Q2_DIR / '2-method-bias-analysis.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    print("  Saved: 2-method-bias-analysis.json")
    
    print("\n" + "="*60)
    print("Method bias analysis complete!")
    print("="*60)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if pct_weight > rank_weight:
        diff = (pct_weight - rank_weight) * 100
        print(f"\n百分比法比排名法更偏向观众投票 {diff:.1f}个百分点")
        print(f"百分比法：{pct_weight:.1%}的淘汰与纯观众投票一致")
        print(f"排名法：{rank_weight:.1%}的淘汰与纯观众投票一致")
    else:
        diff = (rank_weight - pct_weight) * 100
        print(f"\n排名法比百分比法更偏向观众投票 {diff:.1f}个百分点")
        print(f"排名法：{rank_weight:.1%}的淘汰与纯观众投票一致")
        print(f"百分比法：{pct_weight:.1%}的淘汰与纯观众投票一致")
    
    print(f"\n统计检验结论：{test_results['conclusion']}")

if __name__ == '__main__':
    main()
