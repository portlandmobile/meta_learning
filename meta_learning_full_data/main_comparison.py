"""
Main Comparison Script for EMBS Dataset
======================================

This script runs a comprehensive comparison between:
1. Logistic Regression (baseline)
2. Simple Meta Neural Network
3. Enhanced Meta Neural Network

Using the full EMBS dataset with origination and performance data.
"""

import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_loader import DataConfig, EMBSDataLoader
from feature_engineering import FeatureEngineer, create_business_features
from models import LogisticRegressionModel, SimpleMetaNN, EnhancedMetaNN
from business_evaluation import BusinessEvaluator

def main():
    """Main comparison function"""
    print("🚀 EMBS DATASET META LEARNING COMPARISON")
    print("="*80)
    print("Comparing Logistic Regression vs Meta Neural Networks")
    print("Using full EMBS dataset with origination and performance data")
    print("="*80)
    
    # Configuration
    config = DataConfig()
    evaluator = BusinessEvaluator()
    
    # Step 1: Load and preprocess data
    print("\n📊 STEP 1: LOADING AND PREPROCESSING DATA")
    print("-" * 50)
    
    loader = EMBSDataLoader(config)
    orig_data, perf_data, merged_data = loader.load_and_preprocess()
    
    print(f"\n📋 Dataset Summary:")
    print(f"Total loans: {len(merged_data):,}")
    print(f"Default rate: {merged_data['Default'].mean():.1%}")
    print(f"Origination features: {len(orig_data.columns)}")
    print(f"Performance features: {len(perf_data.columns)}")
    
    # Step 2: Feature engineering
    print("\n🔧 STEP 2: FEATURE ENGINEERING")
    print("-" * 50)
    
    # Create business features
    merged_data = create_business_features(merged_data)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X = merged_data.drop('Default', axis=1)
    y = merged_data['Default']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state, stratify=y
    )
    
    print(f"Training set: {len(X_train):,} loans")
    print(f"Test set: {len(X_test):,} loans")
    print(f"Training default rate: {y_train.mean():.1%}")
    print(f"Test default rate: {y_test.mean():.1%}")
    
    # Feature engineering
    engineer = FeatureEngineer()
    X_train_orig, X_test_orig, X_train_perf, X_test_perf = engineer.fit_transform_features(
        X_train, X_test
    )
    
    print(f"\n✅ Feature engineering complete:")
    print(f"Origination features: {X_train_orig.shape[1]}")
    print(f"Performance features: {X_train_perf.shape[1]}")
    
    # Step 3: Train models
    print("\n🧠 STEP 3: TRAINING MODELS")
    print("-" * 50)
    
    model_results = []
    
    # 1. Logistic Regression
    print("\n📊 Training Logistic Regression...")
    start_time = time.time()
    
    lr_model = LogisticRegressionModel(random_state=config.random_state)
    lr_model.fit(X_train_orig, X_train_perf, y_train.values)
    lr_pred = lr_model.predict_proba(X_test_orig, X_test_perf)
    
    lr_time = time.time() - start_time
    print(f"✅ Logistic Regression trained in {lr_time:.2f} seconds")
    
    # Evaluate LR
    lr_result = evaluator.evaluate_model("Logistic Regression", y_test.values, lr_pred)
    model_results.append(lr_result)
    
    # 2. Simple Meta Neural Network
    print("\n🧠 Training Simple Meta Neural Network...")
    start_time = time.time()
    
    simple_model = SimpleMetaNN(
        orig_dim=X_train_orig.shape[1], 
        perf_dim=X_train_perf.shape[1],
        random_state=config.random_state
    )
    simple_model.fit(X_train_orig, X_train_perf, y_train.values, epochs=100)
    simple_pred = simple_model.predict_proba(X_test_orig, X_test_perf)
    
    simple_time = time.time() - start_time
    print(f"✅ Simple Meta NN trained in {simple_time:.2f} seconds")
    
    # Evaluate Simple Meta NN
    simple_result = evaluator.evaluate_model("Simple Meta NN", y_test.values, simple_pred)
    model_results.append(simple_result)
    
    # 3. Enhanced Meta Neural Network
    print("\n🚀 Training Enhanced Meta Neural Network...")
    start_time = time.time()
    
    enhanced_model = EnhancedMetaNN(
        orig_dim=X_train_orig.shape[1], 
        perf_dim=X_train_perf.shape[1],
        random_state=config.random_state
    )
    enhanced_model.fit(X_train_orig, X_train_perf, y_train.values, epochs=150)
    enhanced_pred = enhanced_model.predict_proba(X_test_orig, X_test_perf)
    
    enhanced_time = time.time() - start_time
    print(f"✅ Enhanced Meta NN trained in {enhanced_time:.2f} seconds")
    
    # Evaluate Enhanced Meta NN
    enhanced_result = evaluator.evaluate_model("Enhanced Meta NN", y_test.values, enhanced_pred)
    model_results.append(enhanced_result)
    
    # Step 4: Model comparison
    print("\n📊 STEP 4: MODEL COMPARISON")
    print("-" * 50)
    
    # Compare models
    comparison_df = evaluator.compare_models(model_results)
    
    # Business impact analysis
    business_impact = evaluator.calculate_business_impact(model_results)
    
    # Step 5: Results summary
    print("\n🎉 FINAL RESULTS SUMMARY")
    print("="*80)
    
    print(f"\n⏱️  Training Times:")
    print(f"Logistic Regression: {lr_time:.2f} seconds")
    print(f"Simple Meta NN: {simple_time:.2f} seconds")
    print(f"Enhanced Meta NN: {enhanced_time:.2f} seconds")
    
    print(f"\n💰 Best Model: {business_impact['best_model']}")
    print(f"Annual Profit Improvement: ${business_impact['annual_improvement']:,.0f}")
    print(f"Improvement Percentage: {business_impact['profit_improvement_pct']:.2f}%")
    
    print(f"\n📈 Key Insights:")
    print(f"• Dataset size: {len(merged_data):,} loans")
    print(f"• Default rate: {merged_data['Default'].mean():.1%}")
    print(f"• Feature engineering: {X_train_orig.shape[1]} origination + {X_train_perf.shape[1]} performance features")
    print(f"• Best model profit: ${comparison_df.iloc[0]['Profit']:,.0f}")
    print(f"• Best model recall: {comparison_df.iloc[0]['Recall']:.1%}")
    
    # Step 6: Save results
    print(f"\n💾 SAVING RESULTS")
    print("-" * 50)
    
    # Save comparison results
    comparison_df.to_csv('model_comparison_results.csv', index=False)
    print("✅ Saved model comparison results to 'model_comparison_results.csv'")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'y_true': y_test.values,
        'lr_pred': lr_pred,
        'simple_pred': simple_pred,
        'enhanced_pred': enhanced_pred
    })
    predictions_df.to_csv('model_predictions.csv', index=False)
    print("✅ Saved model predictions to 'model_predictions.csv'")
    
    print(f"\n🎯 COMPARISON COMPLETE!")
    print("="*80)
    
    return model_results, comparison_df, business_impact

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    # Run main comparison
    model_results, comparison_df, business_impact = main()
    
    print(f"\n✅ All done! Check the generated CSV files for detailed results.")



