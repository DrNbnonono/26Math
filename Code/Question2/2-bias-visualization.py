"""
Question 2: Method Bias Visualization
Create visualization for method bias toward fan votes
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Use English labels by default
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path('e:/Competition/数学建模/26美赛')
Q2_DIR = BASE_DIR / 'Data' / 'Question2'
FIG_DIR = BASE_DIR / 'Latex' / 'figures'

def load_bias_data():
    """Load bias analysis results"""
    with open(Q2_DIR / '2-method-bias-analysis.json', 'r') as f:
        return json.load(f)

def create_bias_comparison_chart(bias_data):
    """Create comprehensive bias comparison visualization"""
    print("Creating method bias comparison chart...")
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Extract data
    pct_alignment = bias_data['fan_vote_alignment']['percentage_method']
    rank_alignment = bias_data['fan_vote_alignment']['rank_method']
    
    pct_bias_scores = [r['bias_score'] for r in bias_data['bias_scores']['percentage_method']]
    rank_bias_scores = [r['bias_score'] for r in bias_data['bias_scores']['rank_method']]
    
    # 1. Bar chart - Fan vote alignment comparison
    ax1 = fig.add_subplot(gs[0, :])
    methods = ['Percentage Method', 'Rank Method']
    alignments = [pct_alignment * 100, rank_alignment * 100]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = ax1.bar(methods, alignments, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, val in zip(bars, alignments):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Add difference annotation
    diff = alignments[0] - alignments[1]
    ax1.annotate('', xy=(0, alignments[0]), xytext=(1, alignments[1]),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax1.text(0.5, (alignments[0] + alignments[1])/2,
            f'Difference: {diff:.1f}%',
            ha='center', va='center', fontsize=12, 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax1.set_ylabel('Alignment with Fan Voting (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Bias Toward Fan Votes (Method Comparison)', fontsize=15, fontweight='bold', pad=20)
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% Baseline')
    ax1.legend(fontsize=10)
    
    # 2. Distribution of bias scores - Percentage method
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(pct_bias_scores, bins=20, color='#FF6B6B', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(pct_bias_scores), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(pct_bias_scores):.3f}')
    ax2.set_xlabel('Bias Score', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Percentage Method Bias Score Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    
    # 3. Distribution of bias scores - Rank method
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(rank_bias_scores, bins=20, color='#4ECDC4', alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(rank_bias_scores), color='blue', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(rank_bias_scores):.3f}')
    ax3.set_xlabel('Bias Score', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Rank Method Bias Score Distribution', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    
    # 4. Box plot comparison
    ax4 = fig.add_subplot(gs[2, 0])
    data_to_plot = [pct_bias_scores, rank_bias_scores]
    bp = ax4.boxplot(data_to_plot, tick_labels=['Percentage', 'Rank'],
                     patch_artist=True, widths=0.6)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], ['#FF6B6B', '#4ECDC4']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_ylabel('Bias Score', fontsize=11)
    ax4.set_title('Bias Score Boxplot Comparison', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 5. Statistical test results
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    # Create text summary
    test_results = bias_data['statistical_test']
    summary_text = f"""
    Statistical Test (t-test)
    ━━━━━━━━━━━━━━━━━━━━━━
    
    Percentage mean: {test_results['percentage_mean']:.4f}
    Rank mean: {test_results['rank_mean']:.4f}
    
    t-statistic: {test_results['t_statistic']:.4f}
    p-value: {test_results['p_value']:.4f}
    
    Significance: p < 0.01 **
    
    Conclusion:
    {test_results['conclusion']}
    
    ━━━━━━━━━━━━━━━━━━━━━━
    ** Significant at 1% level
    """
    
    ax5.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('DWTS Voting Method Bias Analysis', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(FIG_DIR / '2-fig-method-bias-comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 2-fig-method-bias-comparison.png")

def create_alignment_pie_chart(bias_data):
    """Create pie chart showing alignment breakdown"""
    print("Creating alignment pie chart...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    pct_alignment = bias_data['fan_vote_alignment']['percentage_method'] * 100
    rank_alignment = bias_data['fan_vote_alignment']['rank_method'] * 100
    
    # Percentage method
    ax1 = axes[0]
    sizes1 = [pct_alignment, 100 - pct_alignment]
    labels1 = [f'Aligns with Fan Vote\n{pct_alignment:.1f}%', f'Diverges\n{100-pct_alignment:.1f}%']
    colors1 = ['#FF6B6B', '#FFE5E5']
    explode1 = (0.1, 0)
    
    wedges1, texts1, autotexts1 = ax1.pie(sizes1, explode=explode1, labels=labels1, colors=colors1,
                                           autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
    for autotext in autotexts1:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(14)
    
    ax1.set_title('Percentage Method vs Pure Fan Voting', fontsize=13, fontweight='bold', pad=20)
    
    # Rank method
    ax2 = axes[1]
    sizes2 = [rank_alignment, 100 - rank_alignment]
    labels2 = [f'Aligns with Fan Vote\n{rank_alignment:.1f}%', f'Diverges\n{100-rank_alignment:.1f}%']
    colors2 = ['#4ECDC4', '#E5F9F7']
    explode2 = (0.1, 0)
    
    wedges2, texts2, autotexts2 = ax2.pie(sizes2, explode=explode2, labels=labels2, colors=colors2,
                                           autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
    for autotext in autotexts2:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(14)
    
    ax2.set_title('Rank Method vs Pure Fan Voting', fontsize=13, fontweight='bold', pad=20)
    
    plt.suptitle('Alignment with Pure Fan Voting (Method Comparison)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_DIR / '2-fig-alignment-pie-chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 2-fig-alignment-pie-chart.png")

def main():
    print("="*60)
    print("Question 2: Method Bias Visualization")
    print("="*60)
    
    # Load data
    bias_data = load_bias_data()
    
    # Create visualizations
    create_bias_comparison_chart(bias_data)
    create_alignment_pie_chart(bias_data)
    
    print("\n" + "="*60)
    print("Method bias visualizations complete!")
    print("="*60)

if __name__ == '__main__':
    main()
