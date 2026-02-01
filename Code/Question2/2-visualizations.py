"""
Question 2: Visualization Generation
Generate 4 visualization figures for Question 2
"""

import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path('e:/Competition/数学建模/26美赛')
Q2_DIR = BASE_DIR / 'Data' / 'Question2'
FIG_DIR = BASE_DIR / 'Latex' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load all Q2 results"""
    with open(Q2_DIR / '2-unified-data.pkl', 'rb') as f:
        unified = pickle.load(f)
    
    comparison = pd.read_csv(Q2_DIR / '2-method-comparison-results.csv')
    
    with open(Q2_DIR / '2-divergence-analysis.json', 'r') as f:
        divergence = json.load(f)
    
    metrics = pd.read_csv(Q2_DIR / '2-contestant-metrics.csv')
    
    with open(Q2_DIR / '2-case-study-results.json', 'r') as f:
        cases = json.load(f)
    
    return unified, comparison, divergence, metrics, cases

def fig1_method_divergence(comparison, divergence):
    """Figure 1: Method divergence heatmap"""
    print("Generating Figure 1: Method Divergence...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create divergence matrix
    seasons = comparison['season'].values
    diverged = comparison['diverged'].astype(int).values
    divergence_weeks = comparison['divergence_week'].fillna(0).values
    
    # Bar plot
    colors = ['green' if not d else 'red' for d in diverged]
    ax.bar(seasons, [1]*len(seasons), color=colors, alpha=0.7, edgecolor='black')
    
    # Add divergence week labels
    for i, (s, d, w) in enumerate(zip(seasons, diverged, divergence_weeks)):
        if d:
            ax.text(s, 0.5, f'W{int(w)}', ha='center', va='center', fontsize=8, color='white', weight='bold')
    
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Method Divergence', fontsize=12)
    ax.set_title('Percentage vs Rank Method Divergence Across Seasons', fontsize=14, weight='bold')
    ax.set_ylim(0, 1.2)
    ax.set_xticks(seasons)
    ax.grid(axis='y', alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.7, label='Diverged'),
                      Patch(facecolor='green', alpha=0.7, label='No Divergence')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / '2-fig-method-divergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 2-fig-method-divergence.png")

def fig2_controversial_trajectory(unified, cases):
    """Figure 2: Weekly trajectory for 4 controversial cases"""
    print("Generating Figure 2: Controversial Contestant Trajectories...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, case in enumerate(cases):
        name = case['name']
        season = case['season']
        
        # Get contestant data
        contestant_data = unified[(unified['celebrity_name'] == name) & 
                                 (unified['season'] == season)].sort_values('week')
        
        ax = axes[idx]
        
        # Plot judge rank and fan vote
        weeks = contestant_data['week'].values
        judge_ranks = contestant_data['judge_rank'].values
        fan_votes = contestant_data['fan_vote_percent'].values * 100  # Convert to percentage
        
        ax2 = ax.twinx()
        
        line1 = ax.plot(weeks, judge_ranks, 'o-', color='blue', linewidth=2, markersize=6, label='Judge Rank')
        line2 = ax2.plot(weeks, fan_votes, 's-', color='red', linewidth=2, markersize=6, label='Fan Vote %')
        
        ax.set_xlabel('Week', fontsize=10)
        ax.set_ylabel('Judge Rank (lower is better)', fontsize=10, color='blue')
        ax2.set_ylabel('Fan Vote Percentage (%)', fontsize=10, color='red')
        ax.set_title(f'{name} (Season {season}, Placement: {case["final_placement"]})', 
                    fontsize=11, weight='bold')
        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        ax.grid(alpha=0.3)
        ax.invert_yaxis()  # Lower rank is better
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / '2-fig-controversial-trajectory.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 2-fig-controversial-trajectory.png")

def fig3_cluster_analysis(metrics):
    """Figure 3: K-means clustering results"""
    print("Generating Figure 3: Cluster Analysis...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    for cluster_label in ['Fan-type', 'Balanced', 'Judge-type']:
        cluster_data = metrics[metrics['cluster_label'] == cluster_label]
        ax.scatter(cluster_data['avg_judge_rank'], 
                  cluster_data['avg_fan_vote'] * 100,
                  label=cluster_label, alpha=0.6, s=50)
    
    ax.set_xlabel('Average Judge Rank', fontsize=12)
    ax.set_ylabel('Average Fan Vote Percentage (%)', fontsize=12)
    ax.set_title('Contestant Clustering: Fan-type vs Judge-type vs Balanced', 
                fontsize=14, weight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.invert_xaxis()  # Lower rank is better
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / '2-fig-cluster-analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 2-fig-cluster-analysis.png")

def fig4_rule_comparison(metrics):
    """Figure 4: Bottom-two rule impact analysis"""
    print("Generating Figure 4: Rule Comparison...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Distribution of deviation index by cluster
    ax1 = axes[0]
    cluster_order = ['Judge-type', 'Balanced', 'Fan-type']
    data_to_plot = [metrics[metrics['cluster_label'] == label]['deviation_index'].values 
                    for label in cluster_order]
    
    bp = ax1.boxplot(data_to_plot, labels=cluster_order, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightcoral']):
        patch.set_facecolor(color)
    
    ax1.set_ylabel('Deviation Index', fontsize=12)
    ax1.set_title('Deviation Index Distribution by Cluster', fontsize=13, weight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Right: Final placement distribution
    ax2 = axes[1]
    for cluster_label, color in zip(['Fan-type', 'Balanced', 'Judge-type'], 
                                    ['red', 'green', 'blue']):
        cluster_data = metrics[metrics['cluster_label'] == cluster_label]
        placements = cluster_data['final_placement'].dropna()
        ax2.hist(placements, bins=20, alpha=0.5, label=cluster_label, color=color)
    
    ax2.set_xlabel('Final Placement', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Final Placement Distribution by Cluster', fontsize=13, weight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / '2-fig-rule-comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 2-fig-rule-comparison.png")

def main():
    print("="*60)
    print("Question 2: Visualization Generation")
    print("="*60)
    
    # Load data
    unified, comparison, divergence, metrics, cases = load_data()
    
    # Generate figures
    fig1_method_divergence(comparison, divergence)
    fig2_controversial_trajectory(unified, cases)
    fig3_cluster_analysis(metrics)
    fig4_rule_comparison(metrics)
    
    print("\n" + "="*60)
    print("All Question 2 visualizations complete!")
    print("="*60)
    print("\nGenerated figures:")
    print("  - 2-fig-method-divergence.png")
    print("  - 2-fig-controversial-trajectory.png")
    print("  - 2-fig-cluster-analysis.png")
    print("  - 2-fig-rule-comparison.png")

if __name__ == '__main__':
    main()
