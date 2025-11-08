#!/usr/bin/env python3
"""
Risk-Optimized Model Comparison
Optimize for detecting risky loans (high recall) instead of pure profit
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import time
from data_loader import DataConfig, EMBSDataLoader
from feature_engineering import FeatureEngineer, create_business_features
from models import SimpleMetaNN, EnhancedMetaNN

def optimize_for_recall(y_true, y_pred_proba, target_recall=0.8):
    """
    Find threshold that achieves target recall (default detection)
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    best_recall = 0
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        if recall >= target_recall and recall > best_recall:
            best_recall = recall
            best_threshold = threshold
    
    return best_threshold, best_recall

def evaluate_model_risk_optimized(model_name, y_true, y_pred_proba, target_recall=0.8):
    """
    Evaluate model optimized for risk detection (recall)
    """
    print(f"\n🎯 {model_name} — Risk-Optimized (Target Recall: {target_recall:.0%})")
    print("-" * 60)
    
    # Find threshold for target recall
    threshold, achieved_recall = optimize_for_recall(y_true, y_pred_proba, target_recall)
    
    # Predict with optimal threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    
    print(f"Threshold: {threshold:.4f} | Achieved Recall: {recall:.1%}")
    print(f"Accuracy:  {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    print(f"Confusion  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
    
    # Business impact with risk-optimized approach
    # Assume: revenue per approved loan, loss per default
    revenue_per_loan = 16250  # From previous analysis
    loss_per_default = 23040  # From previous analysis
    
    total_loans = len(y_true)
    approved_loans = tn + fn  # True negatives + False negatives (approved but defaulted)
    rejected_loans = fp + tp  # False positives + True positives
    
    revenue = approved_loans * revenue_per_loan
    loss = fn * loss_per_default  # False negatives = approved loans that defaulted
    profit = revenue - loss
    
    print(f"\n💰 Business Impact (Risk-Optimized):")
    print(f"Approved loans: {approved_loans:,} ({approved_loans/total_loans:.1%})")
    print(f"Rejected loans: {rejected_loans:,} ({rejected_loans/total_loans:.1%})")
    print(f"Revenue: ${revenue:,.0f} | Loss: ${loss:,.0f} | Profit: ${profit:,.0f}")
    
    return {
        'model': model_name,
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'profit': profit,
        'approved_loans': approved_loans,
        'rejected_loans': rejected_loans,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
    }

def main():
    print("🚀 RISK-OPTIMIZED MODEL COMPARISON")
    print("=" * 80)
    print("Optimizing for Default Detection (High Recall) Instead of Pure Profit")
    print("=" * 80)
    
    # Load and preprocess data
    print("\n📊 STEP 1: LOADING AND PREPROCESSING DATA")
    print("-" * 50)
    
    config = DataConfig()
    loader = EMBSDataLoader(config)
    orig_data, perf_data, merged_data = loader.load_and_preprocess()
    
    print(f"\nDataset loaded: {len(merged_data):,} loans")
    print(f"Default rate: {merged_data['Default'].mean():.3%}")
    
    # Create business features
    merged_data = create_business_features(merged_data)
    
    # Feature engineering
    print("\n🔧 STEP 2: FEATURE ENGINEERING")
    print("-" * 50)
    
    X = merged_data.drop('Default', axis=1)
    y = merged_data['Default']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    engineer = FeatureEngineer()
    X_train_orig, X_test_orig, X_train_perf, X_test_perf = engineer.fit_transform_features(
        X_train, X_test
    )
    
    print(f"Training set: {len(X_train):,} loans")
    print(f"Test set: {len(X_test):,} loans")
    print(f"Test default rate: {y_test.mean():.3%}")
    
    # Train models with risk-optimized parameters
    print("\n🤖 STEP 3: TRAINING RISK-OPTIMIZED MODELS")
    print("-" * 50)
    
    results = []
    
    # 1. Logistic Regression (with class weights for recall)
    print("\n📊 Training Logistic Regression (Class-Weighted)...")
    start_time = time.time()
    
    lr = LogisticRegression(
        class_weight='balanced',  # Balance classes for better recall
        max_iter=2000,
        random_state=42
    )
    
    # Combine features for logistic regression
    X_train_combined = np.hstack([X_train_orig, X_train_perf])
    X_test_combined = np.hstack([X_test_orig, X_test_perf])
    
    lr.fit(X_train_combined, y_train)
    lr_proba = lr.predict_proba(X_test_combined)[:, 1]
    
    lr_time = time.time() - start_time
    print(f"✅ LR trained in {lr_time:.2f} seconds")
    
    # 2. Simple Meta NN (with focal loss for recall)
    print("\n📊 Training Simple Meta NN (Focal Loss)...")
    start_time = time.time()
    
    # Create focal loss for better recall
    def focal_loss(alpha=0.25, gamma=2.0):
        def focal_loss_fixed(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
            
            # Calculate focal loss
            alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
            p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            focal_weight = alpha_t * tf.pow((1 - p_t), gamma)
            
            focal_loss = -focal_weight * tf.math.log(p_t)
            return tf.reduce_mean(focal_loss)
        return focal_loss_fixed
    
    simple_meta = SimpleMetaNN(
        orig_dim=X_train_orig.shape[1],
        perf_dim=X_train_perf.shape[1],
        random_state=42
    )
    
    # Build and compile with focal loss
    model = simple_meta.build_model()
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=focal_loss(alpha=0.75, gamma=2.0),  # Higher alpha for recall
        metrics=['accuracy']
    )
    
    # Train with early stopping
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6)
    ]
    
    model.fit(
        [X_train_orig, X_train_perf], y_train,
        epochs=100,
        batch_size=256,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    simple_proba = model.predict([X_test_orig, X_test_perf], verbose=0).ravel()
    simple_time = time.time() - start_time
    print(f"✅ Simple Meta NN trained in {simple_time:.2f} seconds")
    
    # 3. Enhanced Meta NN (with focal loss)
    print("\n📊 Training Enhanced Meta NN (Focal Loss)...")
    start_time = time.time()
    
    enhanced_meta = EnhancedMetaNN(
        orig_dim=X_train_orig.shape[1],
        perf_dim=X_train_perf.shape[1],
        random_state=42
    )
    
    # Build and compile with focal loss
    enhanced_model = enhanced_meta.build_model()
    enhanced_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=focal_loss(alpha=0.75, gamma=2.0),
        metrics=['accuracy']
    )
    
    enhanced_model.fit(
        [X_train_orig, X_train_perf], y_train,
        epochs=100,
        batch_size=256,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    enhanced_proba = enhanced_model.predict([X_test_orig, X_test_perf], verbose=0).ravel()
    enhanced_time = time.time() - start_time
    print(f"✅ Enhanced Meta NN trained in {enhanced_time:.2f} seconds")
    
    # Evaluate models with risk optimization
    print("\n🎯 STEP 4: RISK-OPTIMIZED EVALUATION")
    print("-" * 50)
    
    # Test different recall targets
    recall_targets = [0.5, 0.7, 0.8, 0.9]
    
    for target_recall in recall_targets:
        print(f"\n{'='*20} TARGET RECALL: {target_recall:.0%} {'='*20}")
        
        # Evaluate each model
        lr_result = evaluate_model_risk_optimized("Logistic Regression", y_test, lr_proba, target_recall)
        simple_result = evaluate_model_risk_optimized("Simple Meta NN", y_test, simple_proba, target_recall)
        enhanced_result = evaluate_model_risk_optimized("Enhanced Meta NN", y_test, enhanced_proba, target_recall)
        
        # Store results
        results.extend([lr_result, simple_result, enhanced_result])
        
        # Compare models
        print(f"\n🏆 RANKING FOR {target_recall:.0%} RECALL TARGET:")
        print("-" * 60)
        model_results = [lr_result, simple_result, enhanced_result]
        model_results.sort(key=lambda x: x['recall'], reverse=True)
        
        for i, result in enumerate(model_results, 1):
            print(f"{i}. {result['model']:<20} | Recall: {result['recall']:.1%} | "
                  f"F1: {result['f1']:.3f} | Profit: ${result['profit']:,.0f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('risk_optimized_results.csv', index=False)
    
    print(f"\n💾 Results saved to 'risk_optimized_results.csv'")
    print("\n🎉 RISK-OPTIMIZED COMPARISON COMPLETE!")

if __name__ == "__main__":
    main()
