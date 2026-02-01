"""
Question 3: Visualization Generation
Generate visualization figures for Question 3
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path('e:/Competition/数学建模/26美赛')
Q3_DIR = BASE_DIR / 'Data' / 'Question3'
FIG_DIR = BASE_DIR / 'Latex' / 'figures'

def load_data():
    """Load model results"""
    judge_importance = pd.read_csv(Q3_DIR / '3-judge-coefficients.csv')
    fan_importance = pd.read_csv(Q3_DIR / '3-fan-coefficients.csv')
    placement_importance = pd.read_csv(Q3_DIR / '3-placement-importance.csv')
    comparison = pd.read_csv(Q3_DIR / '3-influence-comparison.csv')
    
    with open(Q3_DIR / '3-features.pkl', 'rb') as f:
        features = pickle.load(f)
    
    return judge_importance, fan_importance, placement_importance, comparison, features

def fig1_coefficient_comparison(judge_importance, fan_importance):
    """Figure 1: Coefficient comparison between judge and fan models"""
    print("Generating Figure 1: Coefficient Comparison...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Judge model
    ax1 = axes[0]
    top_judge = judge_importance.head(10)
    ax1.barh(range(len(top_judge)), top_judge['importance'], color='steelblue')
    ax1.set_yticks(range(len(top_judge)))
    ax1.set_yticklabels(top_judge['feature'], fontsize=9)
    ax1.set_xlabel('Importance', fontsize=11)
    ax1.set_title('Judge Score Model - Top 10 Features', fontsize=12, weight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Fan model
    ax2 = axes[1]
    top_fan = fan_importance.head(10)
    ax2.barh(range(len(top_fan)), top_fan['importance'], color='coral')
    ax2.set_yticks(range(len(top_fan)))
    ax2.set_yticklabels(top_fan['feature'], fontsize=9)
    ax2.set_xlabel('Importance', fontsize=11)
    ax2.set_title('Fan Vote Model - Top 10 Features', fontsize=12, weight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / '3-fig-coefficient-comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 3-fig-coefficient-comparison.png")

def fig2_feature_importance(placement_importance):
    """Figure 2: Placement model feature importance"""
    print("Generating Figure 2: Feature Importance...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    top_features = placement_importance.head(15)
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
    
    ax.barh(range(len(top_features)), top_features['importance'], color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'], fontsize=10)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('Final Placement Model - Feature Importance', fontsize=14, weight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / '3-fig-feature-importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 3-fig-feature-importance.png")

def fig3_dancer_impact(features):
    """Figure 3: Dancer partner impact on scores and votes"""
    print("Generating Figure 3: Dancer Impact...")
    
    # Calculate average scores by dancer
    dancer_stats = features.groupby('ballroom_partner').agg({
        'judge_avg_score': 'mean',
        'fan_vote_percent': 'mean',
        'celebrity_name': 'count'
    }).reset_index()
    dancer_stats.columns = ['dancer', 'avg_judge_score', 'avg_fan_vote', 'count']
    dancer_stats = dancer_stats[dancer_stats['count'] >= 5]  # At least 5 contestants
    dancer_stats = dancer_stats.sort_values('avg_judge_score', ascending=False).head(20)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(dancer_stats))
    width = 0.35
    
    ax.bar(x - width/2, dancer_stats['avg_judge_score'], width, label='Avg Judge Score', color='steelblue')
    ax.bar(x + width/2, dancer_stats['avg_fan_vote'] * 100, width, label='Avg Fan Vote %', color='coral')
    
    ax.set_xlabel('Professional Dancer', fontsize=11)
    ax.set_ylabel('Score / Percentage', fontsize=11)
    ax.set_title('Professional Dancer Impact on Judge Scores and Fan Votes', fontsize=13, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dancer_stats['dancer'], rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / '3-fig-dancer-impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 3-fig-dancer-impact.png")

def fig4_industry_comparison(features):
    """Figure 4: Industry impact comparison"""
    print("Generating Figure 4: Industry Comparison...")
    
    # Calculate average scores by industry
    industry_stats = features.groupby('industry_clean').agg({
        'judge_avg_score': 'mean',
        'fan_vote_percent': 'mean',
        'celebrity_name': 'count'
    }).reset_index()
    industry_stats.columns = ['industry', 'avg_judge_score', 'avg_fan_vote', 'count']
    industry_stats = industry_stats[industry_stats['count'] >= 10]
    industry_stats = industry_stats.sort_values('avg_fan_vote', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(industry_stats['avg_judge_score'], 
              industry_stats['avg_fan_vote'] * 100,
              s=industry_stats['count'] * 10,
              alpha=0.6,
              c=range(len(industry_stats)),
              cmap='viridis')
    
    for idx, row in industry_stats.iterrows():
        ax.annotate(row['industry'], 
                   (row['avg_judge_score'], row['avg_fan_vote'] * 100),
                   fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Average Judge Score', fontsize=12)
    ax.set_ylabel('Average Fan Vote Percentage (%)', fontsize=12)
    ax.set_title('Industry Impact: Judge Scores vs Fan Votes', fontsize=14, weight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / '3-fig-industry-comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 3-fig-industry-comparison.png")

def fig5_age_effect(features):
    """Figure 5: Age effect on performance"""
    print("Generating Figure 5: Age Effect...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Age vs Judge Score
    ax1 = axes[0]
    age_bins = pd.cut(features['age'], bins=[0, 30, 40, 50, 100])
    age_judge = features.groupby(age_bins)['judge_avg_score'].mean()
    ax1.bar(range(len(age_judge)), age_judge.values, color='steelblue', alpha=0.7)
    ax1.set_xticks(range(len(age_judge)))
    ax1.set_xticklabels(['<30', '30-40', '40-50', '50+'], fontsize=10)
    ax1.set_xlabel('Age Group', fontsize=11)
    ax1.set_ylabel('Average Judge Score', fontsize=11)
    ax1.set_title('Age Effect on Judge Scores', fontsize=12, weight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Age vs Fan Vote
    ax2 = axes[1]
    age_fan = features.groupby(age_bins)['fan_vote_percent'].mean() * 100
    ax2.bar(range(len(age_fan)), age_fan.values, color='coral', alpha=0.7)
    ax2.set_xticks(range(len(age_fan)))
    ax2.set_xticklabels(['<30', '30-40', '40-50', '50+'], fontsize=10)
    ax2.set_xlabel('Age Group', fontsize=11)
    ax2.set_ylabel('Average Fan Vote Percentage (%)', fontsize=11)
    ax2.set_title('Age Effect on Fan Votes', fontsize=12, weight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / '3-fig-age-effect.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 3-fig-age-effect.png")

def main():
    print("="*60)
    print("Question 3: Visualization Generation")
    print("="*60)
    
    # Load data
    judge_importance, fan_importance, placement_importance, comparison, features = load_data()
    
    # Generate figures
    fig1_coefficient_comparison(judge_importance, fan_importance)
    fig2_feature_importance(placement_importance)
    fig3_dancer_impact(features)
    fig4_industry_comparison(features)
    fig5_age_effect(features)
    
    print("\n" + "="*60)
    print("All Question 3 visualizations complete!")
    print("="*60)

if __name__ == '__main__':
    main()
