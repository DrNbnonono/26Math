"""
Question 3: All Models (Judge, Fan, Placement)
Build regression models to analyze factor impacts
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path('e:/Competition/数学建模/26美赛')
Q3_DIR = BASE_DIR / 'Data' / 'Question3'

def load_data():
    """Load preprocessed data"""
    print("Loading data...")
    judge_data = pd.read_csv(Q3_DIR / '3-judge-data.csv')
    fan_data = pd.read_csv(Q3_DIR / '3-fan-data.csv')
    placement_data = pd.read_csv(Q3_DIR / '3-placement-data.csv')
    return judge_data, fan_data, placement_data

def prepare_features(data, target_col, categorical_cols):
    """Prepare features with one-hot encoding"""
    data = data.dropna(subset=[target_col]).copy()
    
    # One-hot encode categorical variables
    data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    # Separate features and target
    exclude_cols = [target_col, 'celebrity_name', 'season', 'week', 
                   'home_state_clean', 'industry', 'home_state', 'home_country']
    feature_cols = [col for col in data_encoded.columns if col not in exclude_cols]
    
    X = data_encoded[feature_cols].select_dtypes(include=[np.number])
    y = data_encoded[target_col]
    
    # Get final feature column names
    feature_cols = X.columns.tolist()
    
    return X, y, feature_cols

def model_judge_scores(judge_data):
    """Model A: Judge score prediction using only celebrity features"""
    print("\n" + "="*60)
    print("Model A: Judge Score Prediction (Celebrity Features Only)")
    print("="*60)
    
    # Prepare data - EXCLUDE judge scores from features
    data = judge_data.dropna(subset=['judge_avg_score']).copy()
    
    # One-hot encode categorical variables
    categorical_cols = ['ballroom_partner', 'industry_clean', 'home_country_clean']
    data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    # Select ONLY celebrity features (age + encoded categoricals)
    # EXCLUDE judge_total, judge_avg_score, and non-feature columns
    exclude_cols = ['judge_avg_score', 'judge_total', 'celebrity_name', 'season', 'week']
    feature_cols = [col for col in data_encoded.columns if col not in exclude_cols]
    
    # Keep all numeric features (age) and all one-hot encoded features
    X = data_encoded[feature_cols]
    
    # Convert boolean columns to int
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
    
    # Ensure all columns are numeric
    X = X.select_dtypes(include=[np.number])
    y = data_encoded['judge_avg_score']
    
    feature_cols = X.columns.tolist()
    
    print(f"  Using {len(feature_cols)} features (age + {len(feature_cols)-1} categorical dummies)")
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X, y)
    
    # Evaluate
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"\n  Top 10 Important Features:")
    print(importance_df.head(10).to_string(index=False))
    
    # Save results
    with open(Q3_DIR / '3-model-judge.pkl', 'wb') as f:
        pickle.dump(model, f)
    importance_df.to_csv(Q3_DIR / '3-judge-coefficients.csv', index=False)
    
    return model, importance_df, {'r2': r2, 'rmse': rmse}

def model_fan_votes(fan_data):
    """Model B: Fan vote prediction using celebrity features"""
    print("\n" + "="*60)
    print("Model B: Fan Vote Prediction (Celebrity Features Only)")
    print("="*60)
    
    # Prepare data - same features as judge model
    data = fan_data.dropna(subset=['fan_vote_percent']).copy()
    
    # One-hot encode categorical variables
    categorical_cols = ['ballroom_partner', 'industry_clean', 'home_country_clean']
    data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    # Select ONLY celebrity features (age + encoded categoricals)
    exclude_cols = ['fan_vote_percent', 'celebrity_name', 'season', 'week']
    feature_cols = [col for col in data_encoded.columns if col not in exclude_cols]
    
    # Keep all numeric features (age) and all one-hot encoded features
    X = data_encoded[feature_cols]
    
    # Convert boolean columns to int
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
    
    # Ensure all columns are numeric
    X = X.select_dtypes(include=[np.number])
    y = data_encoded['fan_vote_percent']
    
    feature_cols = X.columns.tolist()
    
    print(f"  Using {len(feature_cols)} features (age + {len(feature_cols)-1} categorical dummies)")
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X, y)
    
    # Evaluate
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"\n  Top 10 Important Features:")
    print(importance_df.head(10).to_string(index=False))
    
    # Save results
    with open(Q3_DIR / '3-model-fan.pkl', 'wb') as f:
        pickle.dump(model, f)
    importance_df.to_csv(Q3_DIR / '3-fan-coefficients.csv', index=False)
    
    return model, importance_df, {'r2': r2, 'rmse': rmse}

def model_placement(placement_data):
    """Model C: Final placement prediction"""
    print("\n" + "="*60)
    print("Model C: Final Placement Prediction")
    print("="*60)
    
    # Prepare data
    categorical_cols = ['ballroom_partner', 'industry_clean', 'home_country_clean']
    X, y, feature_cols = prepare_features(placement_data, 'final_placement', categorical_cols)
    
    # Train model (GradientBoosting for better performance)
    model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X, y)
    
    # Evaluate
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"\n  Top 10 Important Features:")
    print(importance_df.head(10).to_string(index=False))
    
    # Save results
    with open(Q3_DIR / '3-model-placement.pkl', 'wb') as f:
        pickle.dump(model, f)
    importance_df.to_csv(Q3_DIR / '3-placement-importance.csv', index=False)
    
    return model, importance_df, {'r2': r2, 'rmse': rmse}

def compare_influences(judge_importance, fan_importance):
    """Compare factor impacts between judge scores and fan votes"""
    print("\n" + "="*60)
    print("Comparing Judge vs Fan Influences")
    print("="*60)
    
    # Merge ALL importance scores (not just top 20)
    judge_all = judge_importance.copy()
    judge_all.columns = ['feature', 'judge_importance']
    
    fan_all = fan_importance.copy()
    fan_all.columns = ['feature', 'fan_importance']
    
    comparison = pd.merge(judge_all, fan_all, on='feature', how='outer').fillna(0)
    comparison['difference'] = comparison['fan_importance'] - comparison['judge_importance']
    comparison['abs_difference'] = abs(comparison['difference'])
    
    # Sort by absolute difference to find most divergent features
    comparison = comparison.sort_values('abs_difference', ascending=False)
    
    print("\n  Top 15 Features with LARGEST Impact Differences:")
    print(comparison.head(15)[['feature', 'judge_importance', 'fan_importance', 'difference']].to_string(index=False))
    
    # Categorize features
    print("\n  Feature Categories:")
    
    # Age feature
    age_features = comparison[comparison['feature'].str.contains('age', case=False)]
    if len(age_features) > 0:
        print(f"\n  AGE:")
        print(f"    Judge importance: {age_features['judge_importance'].sum():.4f}")
        print(f"    Fan importance: {age_features['fan_importance'].sum():.4f}")
    
    # Dancer features
    dancer_features = comparison[comparison['feature'].str.contains('ballroom_partner', case=False)]
    if len(dancer_features) > 0:
        print(f"\n  PROFESSIONAL DANCERS (top 5):")
        top_dancers = dancer_features.nlargest(5, 'abs_difference')
        for _, row in top_dancers.iterrows():
            print(f"    {row['feature']}: Judge={row['judge_importance']:.4f}, Fan={row['fan_importance']:.4f}")
    
    # Industry features
    industry_features = comparison[comparison['feature'].str.contains('industry_clean', case=False)]
    if len(industry_features) > 0:
        print(f"\n  INDUSTRIES (top 5):")
        top_industries = industry_features.nlargest(5, 'abs_difference')
        for _, row in top_industries.iterrows():
            print(f"    {row['feature']}: Judge={row['judge_importance']:.4f}, Fan={row['fan_importance']:.4f}")
    
    # Country features
    country_features = comparison[comparison['feature'].str.contains('home_country', case=False)]
    if len(country_features) > 0:
        print(f"\n  HOME COUNTRY/REGION (top 5):")
        top_countries = country_features.nlargest(5, 'abs_difference')
        for _, row in top_countries.iterrows():
            print(f"    {row['feature']}: Judge={row['judge_importance']:.4f}, Fan={row['fan_importance']:.4f}")
    
    # Summary statistics
    print("\n  Summary Statistics:")
    print(f"    Total features compared: {len(comparison)}")
    print(f"    Features favoring judges (judge_importance > fan_importance): {(comparison['judge_importance'] > comparison['fan_importance']).sum()}")
    print(f"    Features favoring fans (fan_importance > judge_importance): {(comparison['fan_importance'] > comparison['judge_importance']).sum()}")
    print(f"    Average absolute difference: {comparison['abs_difference'].mean():.4f}")
    
    # Save comparison
    comparison.to_csv(Q3_DIR / '3-influence-comparison.csv', index=False)
    
    return comparison

def save_summary(judge_metrics, fan_metrics, placement_metrics):
    """Save model performance summary"""
    summary = {
        'judge_model': judge_metrics,
        'fan_model': fan_metrics,
        'placement_model': placement_metrics
    }
    
    with open(Q3_DIR / '3-model-summary.pkl', 'wb') as f:
        pickle.dump(summary, f)
    
    print("\n  Saved: 3-model-summary.pkl")

def main():
    print("="*60)
    print("Question 3: All Models")
    print("="*60)
    
    # Load data
    judge_data, fan_data, placement_data = load_data()
    
    # Model A: Judge scores
    judge_model, judge_importance, judge_metrics = model_judge_scores(judge_data)
    
    # Model B: Fan votes
    fan_model, fan_importance, fan_metrics = model_fan_votes(fan_data)
    
    # Model C: Placement
    placement_model, placement_importance, placement_metrics = model_placement(placement_data)
    
    # Compare influences
    comparison = compare_influences(judge_importance, fan_importance)
    
    # Save summary
    save_summary(judge_metrics, fan_metrics, placement_metrics)
    
    print("\n" + "="*60)
    print("All models complete!")
    print("="*60)
    print("\nModel Performance Summary:")
    print(f"  Judge Score Model R²: {judge_metrics['r2']:.4f}")
    print(f"  Fan Vote Model R²: {fan_metrics['r2']:.4f}")
    print(f"  Placement Model R²: {placement_metrics['r2']:.4f}")

if __name__ == '__main__':
    main()
