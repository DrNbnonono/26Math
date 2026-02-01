"""
Question 2: Identify Controversial Contestants
Identify fan-dependent contestants using technical-fan deviation analysis
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path('e:/Competition/数学建模/26美赛')
Q2_DIR = BASE_DIR / 'Data' / 'Question2'

def load_data():
    """Load preprocessed data"""
    print("Loading data...")
    with open(Q2_DIR / '2-unified-data.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

def calculate_deviation_index(data):
    """
    Calculate technical-fan deviation index for each contestant
    D_i = (1/T_i) * sum((fan_vote - expected_fan_vote) / sigma)
    """
    print("\nCalculating technical-fan deviation index...")
    
    contestant_metrics = []
    
    for (season, contestant), group in data.groupby(['season', 'celebrity_name']):
        # Calculate metrics
        weeks_participated = len(group)
        avg_judge_rank = group['judge_rank'].mean()
        avg_fan_vote = group['fan_vote_percent'].mean()
        
        # Calculate expected fan vote based on judge performance
        # Simple linear regression: fan_vote ~ judge_rank
        season_data = data[data['season'] == season]
        
        # Expected fan vote (inverse relationship with judge rank)
        # Lower rank (better) should get higher fan votes
        max_rank = season_data['judge_rank'].max()
        expected_fan_vote = (max_rank - avg_judge_rank + 1) / max_rank * 0.15  # Rough estimate
        
        # Deviation
        deviation = avg_fan_vote - expected_fan_vote
        
        # Count weeks with judge rank below median (higher rank number = worse)
        weeks_below_median = 0
        for week in group['week'].unique():
            week_data = season_data[season_data['week'] == week]
            median_rank = week_data['judge_rank'].median()
            contestant_rank = group[group['week'] == week]['judge_rank'].iloc[0]
            if contestant_rank > median_rank:
                weeks_below_median += 1
        
        # Final placement
        placement = group['final_placement'].iloc[0] if 'final_placement' in group.columns else None
        
        contestant_metrics.append({
            'season': season,
            'celebrity_name': contestant,
            'weeks_participated': weeks_participated,
            'avg_judge_rank': avg_judge_rank,
            'avg_fan_vote': avg_fan_vote,
            'expected_fan_vote': expected_fan_vote,
            'deviation': deviation,
            'weeks_below_median': weeks_below_median,
            'final_placement': placement,
            'ballroom_partner': group['ballroom_partner'].iloc[0],
            'industry': group['industry'].iloc[0] if 'industry' in group.columns else None
        })
    
    metrics_df = pd.DataFrame(contestant_metrics)
    
    # Calculate deviation index (normalized)
    metrics_df['deviation_index'] = metrics_df['deviation'] / metrics_df['deviation'].std()
    
    print(f"  Calculated metrics for {len(metrics_df)} contestants")
    
    return metrics_df

def identify_controversial_contestants(metrics_df, threshold_percentile=90):
    """
    Identify controversial contestants based on deviation index
    """
    print(f"\nIdentifying controversial contestants (top {100-threshold_percentile}%)...")
    
    # Calculate threshold
    threshold = np.percentile(metrics_df['deviation_index'], threshold_percentile)
    
    # Identify controversial contestants
    controversial = metrics_df[metrics_df['deviation_index'] > threshold].copy()
    controversial = controversial.sort_values('deviation_index', ascending=False)
    
    print(f"  Found {len(controversial)} controversial contestants")
    print(f"  Threshold deviation index: {threshold:.3f}")
    
    # Verify the 4 specified cases
    specified_cases = [
        ('Jerry Rice', 2),
        ('Billy Ray Cyrus', 4),
        ('Bristol Palin', 11),
        ('Bobby Bones', 27)
    ]
    
    print("\n  Verifying specified cases:")
    for name, season in specified_cases:
        match = metrics_df[(metrics_df['celebrity_name'] == name) & 
                          (metrics_df['season'] == season)]
        if len(match) > 0:
            dev_idx = match['deviation_index'].iloc[0]
            is_controversial = dev_idx > threshold
            print(f"    {name} (S{season}): deviation_index={dev_idx:.3f} {'✓ CONTROVERSIAL' if is_controversial else '✗ not controversial'}")
        else:
            print(f"    {name} (S{season}): NOT FOUND in data")
    
    return controversial, threshold

def perform_clustering(metrics_df):
    """
    K-means clustering to categorize contestants into 3 types:
    - Fan-type (high fan votes, low judge scores)
    - Balanced
    - Judge-type (high judge scores, low fan votes)
    """
    print("\nPerforming K-means clustering...")
    
    # Prepare features for clustering
    features = metrics_df[['avg_judge_rank', 'avg_fan_vote', 'deviation_index']].copy()
    features = features.fillna(features.mean())
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # K-means with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    metrics_df['cluster'] = kmeans.fit_predict(features_scaled)
    
    # Identify cluster types based on centroids
    centroids = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=['avg_judge_rank', 'avg_fan_vote', 'deviation_index']
    )
    centroids['cluster'] = range(3)
    
    # Assign labels based on deviation_index
    # Highest deviation = Fan-type, Lowest = Judge-type, Middle = Balanced
    centroids = centroids.sort_values('deviation_index')
    cluster_labels = {
        centroids.iloc[0]['cluster']: 'Judge-type',
        centroids.iloc[1]['cluster']: 'Balanced',
        centroids.iloc[2]['cluster']: 'Fan-type'
    }
    
    metrics_df['cluster_label'] = metrics_df['cluster'].map(cluster_labels)
    
    # Print cluster statistics
    print("\n  Cluster distribution:")
    for label in ['Fan-type', 'Balanced', 'Judge-type']:
        count = (metrics_df['cluster_label'] == label).sum()
        pct = count / len(metrics_df) * 100
        print(f"    {label}: {count} ({pct:.1f}%)")
    
    return metrics_df, cluster_labels

def save_results(metrics_df, controversial_df):
    """Save identification results"""
    print("\nSaving results...")
    
    # Save all contestant metrics
    metrics_df.to_csv(Q2_DIR / '2-contestant-metrics.csv', index=False)
    print("  Saved: 2-contestant-metrics.csv")
    
    # Save controversial contestants
    controversial_df.to_csv(Q2_DIR / '2-controversial-contestants.csv', index=False)
    print("  Saved: 2-controversial-contestants.csv")
    
    # Save clustering results
    cluster_summary = {
        'total_contestants': len(metrics_df),
        'fan_type_count': int((metrics_df['cluster_label'] == 'Fan-type').sum()),
        'balanced_count': int((metrics_df['cluster_label'] == 'Balanced').sum()),
        'judge_type_count': int((metrics_df['cluster_label'] == 'Judge-type').sum()),
        'controversial_count': len(controversial_df),
        'top_controversial': controversial_df.head(10)[['season', 'celebrity_name', 
                                                         'deviation_index', 'final_placement']].to_dict('records')
    }
    
    with open(Q2_DIR / '2-clustering-summary.pkl', 'wb') as f:
        pickle.dump(cluster_summary, f)
    print("  Saved: 2-clustering-summary.pkl")

def main():
    print("="*60)
    print("Question 2: Controversial Contestant Identification")
    print("="*60)
    
    # Load data
    data = load_data()
    
    # Calculate deviation metrics
    metrics_df = calculate_deviation_index(data)
    
    # Identify controversial contestants
    controversial_df, threshold = identify_controversial_contestants(metrics_df)
    
    # Perform clustering
    metrics_df, cluster_labels = perform_clustering(metrics_df)
    
    # Save results
    save_results(metrics_df, controversial_df)
    
    print("\n" + "="*60)
    print("Controversial contestant identification complete!")
    print("="*60)
    print("\nTop 10 most controversial contestants:")
    print(controversial_df.head(10)[['season', 'celebrity_name', 'deviation_index', 
                                     'final_placement']])

if __name__ == '__main__':
    main()
