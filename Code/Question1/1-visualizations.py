"""
Visualization Generation for MCM Problem C Question 1
Generate 5 charts with English labels for the paper
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

OUTPUT_DIR = Path('e:/Competition/数学建模/26美赛')
FIGURES_DIR = OUTPUT_DIR / 'Latex' / 'figures'

def load_data():
    """Load all processed data"""
    percentage_df = pd.read_csv(OUTPUT_DIR / '1-percentage-vote-estimates.csv')
    rank_df = pd.read_csv(OUTPUT_DIR / '1-rank-vote-estimates.csv')
    
    # Combine
    all_results = pd.concat([percentage_df, rank_df], ignore_index=True)
    
    with open(OUTPUT_DIR / '1-percentage-metrics.pkl', 'rb') as f:
        percentage_metrics = pickle.load(f)
    
    with open(OUTPUT_DIR / '1-rank-metrics.pkl', 'rb') as f:
        rank_metrics = pickle.load(f)
    
    return all_results, percentage_metrics, rank_metrics

def fig1_vote_distribution_heatmap(all_results):
    """Figure 1: Vote percentage distribution heatmap across seasons"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Percentage method seasons
    pct_data = all_results[all_results['method'] == 'percentage']
    pct_pivot = pct_data.groupby(['season', 'week'])['fan_vote_percent'].mean().unstack()
    
    sns.heatmap(pct_pivot, cmap='YlOrRd', ax=axes[0], cbar_kws={'label': 'Avg. Fan Vote %'})
    axes[0].set_title('Fan Vote Distribution - Percentage Method\n(Seasons 3-27)', fontsize=12)
    axes[0].set_xlabel('Week')
    axes[0].set_ylabel('Season')
    
    # Rank method seasons
    rank_data = all_results[all_results['method'] == 'rank']
    rank_pivot = rank_data.groupby(['season', 'week'])['fan_vote_percent'].mean().unstack()
    
    sns.heatmap(rank_pivot, cmap='YlGnBu', ax=axes[1], cbar_kws={'label': 'Avg. Fan Vote %'})
    axes[1].set_title('Fan Vote Distribution - Rank Method\n(Seasons 1-2, 28-34)', fontsize=12)
    axes[1].set_xlabel('Week')
    axes[1].set_ylabel('Season')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '1-fig-vote-distribution.png', dpi=300, bbox_inches='tight')
    print("Saved: 1-fig-vote-distribution.png")
    plt.close()

def fig2_smoothness_validation(all_results):
    """Figure 2: Smoothness validation - week-to-week vote changes"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    changes = []
    for (season, contestant), group in all_results.groupby(['season', 'celebrity_name']):
        group_sorted = group.sort_values('week')
        votes = group_sorted['fan_vote_percent'].values
        for i in range(1, len(votes)):
            changes.append({
                'season': season,
                'method': group_sorted.iloc[0]['method'],
                'change': abs(votes[i] - votes[i-1]),
                'week_pair': f"{group_sorted.iloc[i-1]['week']}-{group_sorted.iloc[i]['week']}"
            })
    
    changes_df = pd.DataFrame(changes)
    
    # Distribution of changes
    pct_changes = changes_df[changes_df['method'] == 'percentage']['change']
    rank_changes = changes_df[changes_df['method'] == 'rank']['change']
    
    axes[0].hist(pct_changes, bins=50, alpha=0.7, label='Percentage Method', color='coral')
    axes[0].hist(rank_changes, bins=50, alpha=0.7, label='Rank Method', color='skyblue')
    axes[0].axvline(np.mean(pct_changes), color='red', linestyle='--', label=f'Mean (Pct): {np.mean(pct_changes):.4f}')
    axes[0].axvline(np.mean(rank_changes), color='blue', linestyle='--', label=f'Mean (Rank): {np.mean(rank_changes):.4f}')
    axes[0].set_xlabel('Absolute Vote % Change (Week-to-Week)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Vote Changes Between Consecutive Weeks')
    axes[0].legend()
    
    # Box plot by method
    sns.boxplot(data=changes_df, x='method', y='change', ax=axes[1])
    axes[1].set_xlabel('Voting Method')
    axes[1].set_ylabel('Absolute Vote % Change')
    axes[1].set_title('Vote Change Comparison by Method')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '1-fig-smoothness-validation.png', dpi=300, bbox_inches='tight')
    print("Saved: 1-fig-smoothness-validation.png")
    plt.close()

def fig3_elimination_accuracy(all_results):
    """Figure 3: Elimination prediction accuracy analysis"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Calculate per-season accuracy
    season_accuracy = []
    for season, group in all_results.groupby('season'):
        method = group.iloc[0]['method']
        
        correct = 0
        total = 0
        for (s, w), week_group in group.groupby(['season', 'week']):
            eliminated = week_group[week_group['eliminated'] == True]
            if len(eliminated) > 0:
                total += 1
                # For percentage: lowest combined score should be eliminated
                # For rank: highest combined rank should be eliminated
                if method == 'percentage':
                    min_score = week_group['combined_score'].min()
                    if eliminated['combined_score'].iloc[0] <= min_score + 1e-6:
                        correct += 1
                else:
                    max_rank = week_group['combined_rank'].max()
                    if eliminated['combined_rank'].iloc[0] >= max_rank - 1e-6:
                        correct += 1
        
        season_accuracy.append({
            'season': season,
            'method': method,
            'accuracy': correct / total if total > 0 else 1.0,
            'total_weeks': total
        })
    
    acc_df = pd.DataFrame(season_accuracy)
    
    # Bar plot of accuracy by season
    colors = ['coral' if m == 'percentage' else 'skyblue' for m in acc_df['method']]
    axes[0].bar(acc_df['season'], acc_df['accuracy'], color=colors)
    axes[0].axhline(y=0.75, color='red', linestyle='--', label='75% Threshold')
    axes[0].set_xlabel('Season')
    axes[0].set_ylabel('Elimination Prediction Accuracy')
    axes[0].set_title('Elimination Prediction Accuracy by Season')
    axes[0].legend()
    axes[0].set_ylim([0, 1.1])
    
    # Overall method comparison
    method_acc = acc_df.groupby('method')['accuracy'].mean()
    axes[1].bar(method_acc.index, method_acc.values, color=['skyblue', 'coral'])
    axes[1].set_ylabel('Average Accuracy')
    axes[1].set_title('Overall Method Accuracy Comparison')
    axes[1].set_ylim([0, 1])
    
    for i, v in enumerate(method_acc.values):
        axes[1].text(i, v + 0.02, f'{v:.2%}', ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '1-fig-elimination-accuracy.png', dpi=300, bbox_inches='tight')
    print("Saved: 1-fig-elimination-accuracy.png")
    plt.close()

def fig4_uncertainty_analysis(all_results):
    """Figure 4: Uncertainty analysis - confidence intervals"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Calculate vote percentage range for each contestant across their weeks
    contestant_stats = []
    for (season, contestant), group in all_results.groupby(['season', 'celebrity_name']):
        if len(group) >= 2:
            contestant_stats.append({
                'season': season,
                'contestant': contestant,
                'method': group.iloc[0]['method'],
                'mean_vote': group['fan_vote_percent'].mean(),
                'std_vote': group['fan_vote_percent'].std(),
                'min_vote': group['fan_vote_percent'].min(),
                'max_vote': group['fan_vote_percent'].max(),
                'range': group['fan_vote_percent'].max() - group['fan_vote_percent'].min(),
                'weeks': len(group)
            })
    
    stats_df = pd.DataFrame(contestant_stats)
    
    # Distribution of ranges
    pct_ranges = stats_df[stats_df['method'] == 'percentage']['range']
    rank_ranges = stats_df[stats_df['method'] == 'rank']['range']
    
    axes[0].hist(pct_ranges, bins=30, alpha=0.7, label='Percentage Method', color='coral')
    axes[0].hist(rank_ranges, bins=30, alpha=0.7, label='Rank Method', color='skyblue')
    axes[0].set_xlabel('Vote % Range (Max - Min)')
    axes[0].set_ylabel('Number of Contestants')
    axes[0].set_title('Vote Percentage Variability Across Weeks')
    axes[0].legend()
    
    # Scatter plot: mean vs std
    pct_stats = stats_df[stats_df['method'] == 'percentage']
    rank_stats = stats_df[stats_df['method'] == 'rank']
    
    axes[1].scatter(pct_stats['mean_vote'], pct_stats['std_vote'], 
                    alpha=0.6, label='Percentage Method', color='coral')
    axes[1].scatter(rank_stats['mean_vote'], rank_stats['std_vote'], 
                    alpha=0.6, label='Rank Method', color='skyblue')
    axes[1].set_xlabel('Mean Vote %')
    axes[1].set_ylabel('Standard Deviation of Vote %')
    axes[1].set_title('Vote Stability: Mean vs Variability')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '1-fig-uncertainty-analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: 1-fig-uncertainty-analysis.png")
    plt.close()

def fig5_season_comparison(all_results):
    """Figure 5: Multi-season comparison - judge vs fan votes"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Sample seasons for detailed analysis
    sample_seasons = [3, 10, 27, 28]
    
    for idx, season in enumerate(sample_seasons):
        ax = axes[idx // 2, idx % 2]
        
        season_data = all_results[all_results['season'] == season]
        method = season_data.iloc[0]['method']
        
        # Scatter plot: judge % vs fan %
        ax.scatter(season_data['judge_percent'], season_data['fan_vote_percent'], 
                  c=season_data['week'], cmap='viridis', alpha=0.7, s=50)
        
        # Add diagonal line (equal judge and fan support)
        max_val = max(season_data['judge_percent'].max(), season_data['fan_vote_percent'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Equal Support Line')
        
        ax.set_xlabel('Judge Score %')
        ax.set_ylabel('Fan Vote %')
        ax.set_title(f'Season {season} ({method.title()} Method): Judge vs Fan Support')
        ax.legend()
        
        # Add colorbar
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Week')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '1-fig-season-comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: 1-fig-season-comparison.png")
    plt.close()

def generate_all_visualizations():
    """Generate all 5 visualization figures"""
    print("="*60)
    print("Generating Visualizations")
    print("="*60)
    
    # Ensure figures directory exists
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    all_results, pct_metrics, rank_metrics = load_data()
    print(f"\nLoaded {len(all_results)} records")
    
    # Generate figures
    print("\nGenerating Figure 1: Vote Distribution Heatmap...")
    fig1_vote_distribution_heatmap(all_results)
    
    print("\nGenerating Figure 2: Smoothness Validation...")
    fig2_smoothness_validation(all_results)
    
    print("\nGenerating Figure 3: Elimination Accuracy...")
    fig3_elimination_accuracy(all_results)
    
    print("\nGenerating Figure 4: Uncertainty Analysis...")
    fig4_uncertainty_analysis(all_results)
    
    print("\nGenerating Figure 5: Season Comparison...")
    fig5_season_comparison(all_results)
    
    print("\n" + "="*60)
    print("All visualizations generated!")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("="*60)

if __name__ == '__main__':
    generate_all_visualizations()
