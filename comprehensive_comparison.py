# comprehensive_comparison.py
# ------------------------------------------------------------
# Comprehensive comparison: Original LR, Original Meta NN, Enhanced Meta NN
# - Same data split and preprocessing for fair comparison
# - Business-focused evaluation with profit optimization
# - Detailed performance analysis and sensitivity testing
# ------------------------------------------------------------

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List, Dict
import time

# sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# tensorflow / keras
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping

# Import enhanced model
from improved_meta_nn import (
    EnhancedConfig, build_enhanced_meta_nn, train_enhanced_meta_nn,
    focal_loss, load_data, split_cols, make_numeric_preproc, make_categorical_preproc,
    fit_transform_preprocessors, business_eval, sweep_thresholds, print_summary_block
)

# ------------------------------------------------------------
# Original Config (from notebook)
# ------------------------------------------------------------
@dataclass
class OriginalConfig:
    csv_path: str = "/Users/peekay/Downloads/Loan_default.csv"
    target: str = "Default"
    drop_cols: Tuple[str, ...] = ("LoanID",)
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5

    # Business economics
    revenue_per_good: float = 125_000 * 0.13   # ~16,250
    loss_per_default: float = 144_000 * 0.16   # ~23,040

    # Threshold sweep grid
    threshold_low: float = 0.05
    threshold_high: float = 0.95
    threshold_points: int = 37

    # Logistic Regression hyperparams
    lr_use_class_weight_balanced: bool = False
    lr_solver: str = "liblinear"
    lr_C: float = 1.0
    lr_max_iter: int = 2000

    # Original simple_meta NN hyperparams
    nn_use_class_weight_balanced: bool = False
    nn_epochs: int = 200
    nn_batch_size: int = 256
    nn_val_split: float = 0.2
    nn_patience: int = 12
    nn_lr: float = 1e-3
    nn_dropout: float = 0.3

# ------------------------------------------------------------
# Original Models (from notebook)
# ------------------------------------------------------------
def build_logistic_regression(cfg: OriginalConfig) -> LogisticRegression:
    class_weight = "balanced" if cfg.lr_use_class_weight_balanced else None
    return LogisticRegression(
        solver=cfg.lr_solver,
        C=cfg.lr_C,
        max_iter=cfg.lr_max_iter,
        class_weight=class_weight,
        n_jobs=None if cfg.lr_solver == "liblinear" else -1,
        random_state=cfg.random_state
    )

def build_simple_meta_nn(num_dim: int, cat_dim: int, cfg: OriginalConfig) -> Model:
    in_num = Input(shape=(num_dim,), name="num_input")
    x_num = Dense(64, activation="relu")(in_num)
    x_num = Dropout(cfg.nn_dropout)(x_num)
    x_num = Dense(32, activation="relu")(x_num)

    in_cat = Input(shape=(cat_dim,), name="cat_input")
    x_cat = Dense(64, activation="relu")(in_cat)
    x_cat = Dropout(cfg.nn_dropout)(x_cat)
    x_cat = Dense(32, activation="relu")(x_cat)

    fused = Concatenate(name="fuse")([x_num, x_cat])
    z = Dense(32, activation="relu")(fused)
    z = Dropout(cfg.nn_dropout)(z)
    z = Dense(16, activation="relu")(z)
    out = Dense(1, activation="sigmoid", name="default_risk")(z)

    model = Model(inputs=[in_num, in_cat], outputs=out, name="simple_meta")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.nn_lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

def build_lr_preprocessor(num_cols, cat_cols):
    return ColumnTransformer([
        ("num", make_numeric_preproc(), num_cols),
        ("cat", make_categorical_preproc(), cat_cols)
    ])

# ------------------------------------------------------------
# Training Functions
# ------------------------------------------------------------
def train_logistic_regression(X_train, y_train, num_cols, cat_cols, cfg: OriginalConfig):
    """Train logistic regression model"""
    print("\n" + "="*60)
    print("TRAINING LOGISTIC REGRESSION")
    print("="*60)
    
    start_time = time.time()
    
    lr_pre = build_lr_preprocessor(num_cols, cat_cols)
    lr = build_logistic_regression(cfg)
    lr_pipe = Pipeline([("pre", lr_pre), ("clf", lr)])

    # Cross-validation
    skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
    cv_acc = cross_val_score(lr_pipe, X_train, y_train, cv=skf, scoring="accuracy")
    cv_f1  = cross_val_score(lr_pipe, X_train, y_train, cv=skf, scoring="f1")
    print(f"LR CV {cfg.cv_folds}-fold | Acc: {cv_acc.mean():.4f} ± {cv_acc.std():.4f} | F1: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    # Fit final model
    lr_pipe.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")
    
    return lr_pipe

def train_original_meta_nn(X_train_num, X_train_cat, y_train, num_dim, cat_dim, cfg: OriginalConfig):
    """Train original simple meta neural network"""
    print("\n" + "="*60)
    print("TRAINING ORIGINAL SIMPLE META NN")
    print("="*60)
    
    start_time = time.time()
    
    tf.random.set_seed(cfg.random_state)
    simple_meta = build_simple_meta_nn(num_dim, cat_dim, cfg)

    # Class weights (disabled in original)
    cw = None
    if cfg.nn_use_class_weight_balanced:
        classes = np.unique(y_train)
        counts = np.bincount(y_train)
        total = counts.sum()
        cw = {cls: total / (len(classes) * counts[cls]) for cls in classes}
        print(f"Class weights: {cw}")

    # Training
    es = EarlyStopping(monitor="val_loss", patience=cfg.nn_patience, restore_best_weights=True, verbose=1)
    history = simple_meta.fit(
        [X_train_num, X_train_cat], y_train.values,
        epochs=cfg.nn_epochs,
        batch_size=cfg.nn_batch_size,
        validation_split=cfg.nn_val_split,
        callbacks=[es],
        class_weight=cw,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")
    
    return simple_meta

def train_enhanced_meta_nn_with_timing(X_train_num, X_train_cat, y_train, X_test_num, X_test_cat, 
                                     num_dim, cat_dim, cfg: EnhancedConfig, optimize_hyperparams: bool = True):
    """Train enhanced meta neural network with timing"""
    print("\n" + "="*60)
    print("TRAINING ENHANCED META NN")
    print("="*60)
    
    start_time = time.time()
    
    enhanced_pred, models = train_enhanced_meta_nn(
        X_train_num, X_train_cat, y_train, X_test_num, X_test_cat, 
        num_dim, cat_dim, cfg, optimize_hyperparams=optimize_hyperparams
    )
    
    training_time = time.time() - start_time
    print(f"Total training time: {training_time:.2f} seconds")
    
    return enhanced_pred, models

# ------------------------------------------------------------
# Comprehensive Evaluation
# ------------------------------------------------------------
def comprehensive_evaluation(models_results: Dict, y_test, cfg):
    """Comprehensive evaluation of all models"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*80)
    
    results = {}
    
    # Evaluate each model
    for model_name, (prob, training_time) in models_results.items():
        print(f"\nEvaluating {model_name}...")
        best_result = print_summary_block(model_name, prob, y_test.values, cfg)
        results[model_name] = {
            'probabilities': prob,
            'best_result': best_result,
            'training_time': training_time,
            'auc': roc_auc_score(y_test.values, prob)
        }
    
    # Ranking by business performance
    print("\n" + "="*80)
    print("BUSINESS PERFORMANCE RANKING")
    print("="*80)
    
    ranking_data = []
    for name, result in results.items():
        best = result['best_result']
        ranking_data.append({
            'name': name,
            'profit': best['profit'],
            'threshold': best['threshold'],
            'accuracy': best['accuracy'],
            'precision': best['precision'],
            'recall': best['recall'],
            'f1': best['f1'],
            'auc': result['auc'],
            'training_time': result['training_time']
        })
    
    # Sort by profit
    ranking_data.sort(key=lambda x: x['profit'], reverse=True)
    
    print(f"{'Rank':<4} {'Model':<20} {'Profit ($)':<15} {'Threshold':<10} {'AUC':<8} {'Recall':<8} {'F1':<8} {'Time (s)':<10}")
    print("-" * 95)
    
    for i, data in enumerate(ranking_data, 1):
        print(f"{i:<4} {data['name']:<20} ${data['profit']:<14,.0f} {data['threshold']:<10.4f} "
              f"{data['auc']:<8.4f} {data['recall']:<8.4f} {data['f1']:<8.4f} {data['training_time']:<10.1f}")
    
    # Improvement analysis
    print("\n" + "="*80)
    print("IMPROVEMENT ANALYSIS")
    print("="*80)
    
    if len(ranking_data) >= 2:
        baseline = ranking_data[-1]  # Worst performing model
        best = ranking_data[0]       # Best performing model
        
        profit_improvement = best['profit'] - baseline['profit']
        profit_improvement_pct = (profit_improvement / baseline['profit']) * 100
        
        recall_improvement = best['recall'] - baseline['recall']
        recall_improvement_pct = (recall_improvement / baseline['recall']) * 100
        
        auc_improvement = best['auc'] - baseline['auc']
        
        print(f"Best Model: {best['name']}")
        print(f"Baseline Model: {baseline['name']}")
        print(f"\nProfit Improvement: ${profit_improvement:,.0f} ({profit_improvement_pct:.2f}%)")
        print(f"Recall Improvement: {recall_improvement:.4f} ({recall_improvement_pct:.2f}%)")
        print(f"AUC Improvement: {auc_improvement:.4f}")
        
        # Business impact
        print(f"\nBusiness Impact:")
        print(f"- Additional profit per year: ${profit_improvement:,.0f}")
        print(f"- Better default detection: {recall_improvement:.1%} improvement")
        print(f"- Reduced false negatives: {int(recall_improvement * len(y_test) * y_test.mean()):,} more defaults caught")
    
    return results, ranking_data

# ------------------------------------------------------------
# Sensitivity Analysis
# ------------------------------------------------------------
def sensitivity_analysis(results: Dict, y_test, cfg, model_names: List[str]):
    """Sensitivity analysis for top models"""
    
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS")
    print("="*80)
    
    # Test sensitivity to payoff assumptions
    payoff_scenarios = [
        ("Conservative", 0.8, 1.2),  # Lower revenue, higher loss
        ("Baseline", 1.0, 1.0),      # Current assumptions
        ("Optimistic", 1.2, 0.8),    # Higher revenue, lower loss
    ]
    
    for scenario_name, rev_mult, loss_mult in payoff_scenarios:
        print(f"\n{scenario_name} Scenario (Revenue: {rev_mult:.1f}x, Loss: {loss_mult:.1f}x):")
        print("-" * 60)
        
        scenario_results = []
        for model_name in model_names:
            prob = results[model_name]['probabilities']
            best, _ = sweep_thresholds(
                y_test.values, prob, cfg.threshold_low, cfg.threshold_high, cfg.threshold_points,
                cfg.revenue_per_good * rev_mult, cfg.loss_per_default * loss_mult
            )
            scenario_results.append((model_name, best['profit'], best['threshold']))
        
        scenario_results.sort(key=lambda x: x[1], reverse=True)
        
        for i, (name, profit, threshold) in enumerate(scenario_results, 1):
            print(f"{i}. {name:<20} | Profit: ${profit:>12,.0f} | Threshold: {threshold:.4f}")

# ------------------------------------------------------------
# Main Comparison Function
# ------------------------------------------------------------
def main_comparison():
    """Main comparison function"""
    
    print("="*80)
    print("COMPREHENSIVE MODEL COMPARISON: LR vs ORIGINAL vs ENHANCED META NN")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    cfg_orig = OriginalConfig()
    cfg_enh = EnhancedConfig()
    
    df = load_data(cfg_orig)
    X, y, num_cols, cat_cols = split_cols(df, cfg_orig.target)
    print(f"Dataset: {df.shape} | Default rate={y.mean():.3%}")
    print(f"Numeric: {len(num_cols)} | Categorical: {len(cat_cols)}")

    # Same stratified split for all models
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg_orig.test_size, random_state=cfg_orig.random_state, stratify=y
    )

    # Preprocessing
    num_pre, cat_pre, Xtr_num, Xte_num, Xtr_cat, Xte_cat = fit_transform_preprocessors(
        X_train, X_test, num_cols, cat_cols
    )
    
    # Train all models
    models_results = {}
    
    # 1. Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train, num_cols, cat_cols, cfg_orig)
    lr_prob = lr_model.predict_proba(X_test)[:, 1]
    models_results['Logistic Regression'] = (lr_prob, 0)  # LR training time not tracked separately
    
    # 2. Original Simple Meta NN
    original_nn = train_original_meta_nn(Xtr_num, Xtr_cat, y_train, Xtr_num.shape[1], Xtr_cat.shape[1], cfg_orig)
    original_prob = original_nn.predict([Xte_num, Xte_cat], verbose=0).ravel()
    models_results['Original Simple Meta NN'] = (original_prob, 0)  # Training time not tracked separately
    
    # 3. Enhanced Meta NN
    enhanced_prob, enhanced_models = train_enhanced_meta_nn_with_timing(
        Xtr_num, Xtr_cat, y_train, Xte_num, Xte_cat,
        Xtr_num.shape[1], Xtr_cat.shape[1], cfg_enh, optimize_hyperparams=False  # Skip optimization for speed
    )
    models_results['Enhanced Meta NN'] = (enhanced_prob, 0)  # Training time not tracked separately
    
    # Comprehensive evaluation
    results, ranking_data = comprehensive_evaluation(models_results, y_test, cfg_orig)
    
    # Sensitivity analysis for top 2 models
    top_models = [data['name'] for data in ranking_data[:2]]
    sensitivity_analysis(results, y_test, cfg_orig, top_models)
    
    return results, ranking_data

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    results, ranking = main_comparison()
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print("Check the results above for detailed performance metrics and business impact analysis.")

