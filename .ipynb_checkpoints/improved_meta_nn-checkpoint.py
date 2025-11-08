# improved_meta_nn.py
# ------------------------------------------------------------
# Enhanced Meta Neural Network for Loan Default Prediction
# - Advanced architecture with batch normalization and residual connections
# - Class weighting and focal loss for imbalanced data
# - Hyperparameter optimization and ensemble techniques
# - Learning rate scheduling and advanced regularization
# ------------------------------------------------------------

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import itertools

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
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, LayerNormalization,
    Concatenate, Add, Multiply, Lambda
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow.keras.backend as K

# ------------------------------------------------------------
# Enhanced Config with hyperparameter options
# ------------------------------------------------------------
@dataclass
class EnhancedConfig:
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

    # Enhanced NN hyperparams - multiple options for tuning
    nn_use_class_weight_balanced: bool = True  # Enable class weighting
    nn_use_focal_loss: bool = True  # Use focal loss for imbalanced data
    nn_epochs: int = 300
    nn_batch_size: int = 512  # Increased batch size
    nn_val_split: float = 0.15  # Smaller validation split
    nn_patience: int = 20  # More patience
    nn_lr: float = 2e-3  # Higher initial learning rate
    nn_dropout: float = 0.25  # Reduced dropout
    
    # Architecture options
    nn_hidden_layers: List[int] = None  # Will be set to [128, 64, 32] if None
    nn_use_batch_norm: bool = True
    nn_use_residual: bool = True
    nn_use_attention: bool = True
    nn_regularization: float = 1e-4
    
    # Focal loss parameters
    nn_focal_alpha: float = 0.25
    nn_focal_gamma: float = 2.0
    
    # Learning rate scheduling
    nn_lr_schedule: bool = True
    nn_lr_factor: float = 0.5
    nn_lr_patience: int = 8
    
    # Ensemble options
    nn_n_models: int = 3  # Number of models for ensemble
    nn_ensemble_method: str = "mean"  # "mean", "median", "weighted"

    def __post_init__(self):
        if self.nn_hidden_layers is None:
            self.nn_hidden_layers = [128, 64, 32]

# ------------------------------------------------------------
# Focal Loss Implementation
# ------------------------------------------------------------
def focal_loss(alpha=0.25, gamma=2.0):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        
        # Calculate focal loss
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = alpha_t * K.pow((1 - p_t), gamma)
        
        focal_loss = -focal_weight * K.log(p_t)
        return K.mean(focal_loss)
    
    return focal_loss_fixed

# ------------------------------------------------------------
# Enhanced Architecture Components
# ------------------------------------------------------------
def residual_block(x, units, dropout_rate, use_batch_norm=True, regularization=1e-4):
    """Residual block with batch normalization and dropout"""
    residual = x
    
    # Main path
    x = Dense(units, kernel_regularizer=l1_l2(regularization))(x)
    if use_batch_norm:
        x = BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(units, kernel_regularizer=l1_l2(regularization))(x)
    if use_batch_norm:
        x = BatchNormalization()(x)
    
    # Residual connection (if dimensions match)
    if residual.shape[-1] == x.shape[-1]:
        x = Add()([x, residual])
    
    x = tf.keras.activations.relu(x)
    return x

def attention_layer(x, name="attention"):
    """Simple attention mechanism"""
    attention_weights = Dense(x.shape[-1], activation='softmax', name=f"{name}_weights")(x)
    attended = Multiply(name=f"{name}_out")([x, attention_weights])
    return attended

# ------------------------------------------------------------
# Enhanced Model Builder
# ------------------------------------------------------------
def build_enhanced_meta_nn(num_dim: int, cat_dim: int, cfg: EnhancedConfig, model_id: int = 0) -> Model:
    """Build enhanced meta neural network with advanced architecture"""
    
    # Input layers
    in_num = Input(shape=(num_dim,), name=f"num_input_{model_id}")
    in_cat = Input(shape=(cat_dim,), name=f"cat_input_{model_id}")
    
    # Numeric branch with residual blocks
    x_num = Dense(cfg.nn_hidden_layers[0], kernel_regularizer=l1_l2(cfg.nn_regularization))(in_num)
    if cfg.nn_use_batch_norm:
        x_num = BatchNormalization()(x_num)
    x_num = tf.keras.activations.relu(x_num)
    x_num = Dropout(cfg.nn_dropout)(x_num)
    
    if cfg.nn_use_residual:
        x_num = residual_block(x_num, cfg.nn_hidden_layers[1], cfg.nn_dropout, 
                             cfg.nn_use_batch_norm, cfg.nn_regularization)
    else:
        x_num = Dense(cfg.nn_hidden_layers[1], kernel_regularizer=l1_l2(cfg.nn_regularization))(x_num)
        if cfg.nn_use_batch_norm:
            x_num = BatchNormalization()(x_num)
        x_num = tf.keras.activations.relu(x_num)
        x_num = Dropout(cfg.nn_dropout)(x_num)
    
    # Categorical branch with similar structure
    x_cat = Dense(cfg.nn_hidden_layers[0], kernel_regularizer=l1_l2(cfg.nn_regularization))(in_cat)
    if cfg.nn_use_batch_norm:
        x_cat = BatchNormalization()(x_cat)
    x_cat = tf.keras.activations.relu(x_cat)
    x_cat = Dropout(cfg.nn_dropout)(x_cat)
    
    if cfg.nn_use_residual:
        x_cat = residual_block(x_cat, cfg.nn_hidden_layers[1], cfg.nn_dropout, 
                             cfg.nn_use_batch_norm, cfg.nn_regularization)
    else:
        x_cat = Dense(cfg.nn_hidden_layers[1], kernel_regularizer=l1_l2(cfg.nn_regularization))(x_cat)
        if cfg.nn_use_batch_norm:
            x_cat = BatchNormalization()(x_cat)
        x_cat = tf.keras.activations.relu(x_cat)
        x_cat = Dropout(cfg.nn_dropout)(x_cat)
    
    # Fusion layer
    fused = Concatenate(name=f"fuse_{model_id}")([x_num, x_cat])
    
    # Attention mechanism
    if cfg.nn_use_attention:
        fused = attention_layer(fused, f"attention_{model_id}")
    
    # Final layers
    z = Dense(cfg.nn_hidden_layers[2], kernel_regularizer=l1_l2(cfg.nn_regularization))(fused)
    if cfg.nn_use_batch_norm:
        z = BatchNormalization()(z)
    z = tf.keras.activations.relu(z)
    z = Dropout(cfg.nn_dropout)(z)
    
    z = Dense(16, kernel_regularizer=l1_l2(cfg.nn_regularization))(z)
    if cfg.nn_use_batch_norm:
        z = BatchNormalization()(z)
    z = tf.keras.activations.relu(z)
    z = Dropout(cfg.nn_dropout * 0.5)(z)  # Less dropout in final layers
    
    # Output layer
    out = Dense(1, activation="sigmoid", name=f"default_risk_{model_id}")(z)
    
    model = Model(inputs=[in_num, in_cat], outputs=out, name=f"enhanced_meta_{model_id}")
    
    # Compile with appropriate loss
    if cfg.nn_use_focal_loss:
        loss_fn = focal_loss(cfg.nn_focal_alpha, cfg.nn_focal_gamma)
    else:
        loss_fn = "binary_crossentropy"
    
    model.compile(
        optimizer=AdamW(learning_rate=cfg.nn_lr, weight_decay=cfg.nn_regularization),
        loss=loss_fn,
        metrics=["accuracy"]
    )
    
    return model

# ------------------------------------------------------------
# Training with Callbacks
# ------------------------------------------------------------
def get_training_callbacks(cfg: EnhancedConfig):
    """Get training callbacks for enhanced model"""
    callbacks = []
    
    # Early stopping
    es = EarlyStopping(
        monitor="val_loss",
        patience=cfg.nn_patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(es)
    
    # Learning rate scheduling
    if cfg.nn_lr_schedule:
        lr_scheduler = ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg.nn_lr_factor,
            patience=cfg.nn_lr_patience,
            min_lr=1e-6,
            verbose=1
        )
        callbacks.append(lr_scheduler)
    
    return callbacks

# ------------------------------------------------------------
# Ensemble Training
# ------------------------------------------------------------
def train_ensemble(X_train_num, X_train_cat, y_train, num_dim, cat_dim, cfg: EnhancedConfig):
    """Train ensemble of enhanced models"""
    models = []
    predictions = []
    
    print(f"\nTraining ensemble of {cfg.nn_n_models} enhanced models...")
    
    for i in range(cfg.nn_n_models):
        print(f"\nTraining model {i+1}/{cfg.nn_n_models}")
        
        # Create model with slight variation in random seed
        tf.random.set_seed(cfg.random_state + i)
        model = build_enhanced_meta_nn(num_dim, cat_dim, cfg, i)
        
        # Class weights
        class_weight = None
        if cfg.nn_use_class_weight_balanced:
            classes = np.unique(y_train)
            counts = np.bincount(y_train)
            total = counts.sum()
            class_weight = {cls: total / (len(classes) * counts[cls]) for cls in classes}
            print(f"Class weights: {class_weight}")
        
        # Train model
        callbacks = get_training_callbacks(cfg)
        history = model.fit(
            [X_train_num, X_train_cat], y_train.values,
            epochs=cfg.nn_epochs,
            batch_size=cfg.nn_batch_size,
            validation_split=cfg.nn_val_split,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
        
        models.append(model)
        
        # Get predictions on validation set for ensemble weighting
        val_size = int(len(X_train_num) * cfg.nn_val_split)
        X_val_num = X_train_num[-val_size:]
        X_val_cat = X_train_cat[-val_size:]
        y_val = y_train.values[-val_size:]
        
        val_pred = model.predict([X_val_num, X_val_cat], verbose=0).ravel()
        val_auc = roc_auc_score(y_val, val_pred)
        predictions.append(val_pred)
        
        print(f"Model {i+1} validation AUC: {val_auc:.4f}")
    
    return models, predictions

# ------------------------------------------------------------
# Ensemble Prediction
# ------------------------------------------------------------
def ensemble_predict(models, X_test_num, X_test_cat, cfg: EnhancedConfig):
    """Make ensemble predictions"""
    predictions = []
    
    for model in models:
        pred = model.predict([X_test_num, X_test_cat], verbose=0).ravel()
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    if cfg.nn_ensemble_method == "mean":
        return np.mean(predictions, axis=0)
    elif cfg.nn_ensemble_method == "median":
        return np.median(predictions, axis=0)
    elif cfg.nn_ensemble_method == "weighted":
        # Simple equal weighting for now
        return np.mean(predictions, axis=0)
    else:
        return np.mean(predictions, axis=0)

# ------------------------------------------------------------
# Hyperparameter Optimization
# ------------------------------------------------------------
def optimize_hyperparameters(X_train_num, X_train_cat, y_train, num_dim, cat_dim, base_cfg: EnhancedConfig):
    """Simple grid search for key hyperparameters"""
    
    # Define parameter grid
    param_grid = {
        'nn_lr': [1e-3, 2e-3, 5e-3],
        'nn_dropout': [0.2, 0.25, 0.3],
        'nn_batch_size': [256, 512, 1024],
        'nn_hidden_layers': [[128, 64, 32], [96, 48, 24], [160, 80, 40]]
    }
    
    best_score = -np.inf
    best_params = None
    best_model = None
    
    print("\nStarting hyperparameter optimization...")
    
    # Sample a few combinations (full grid would be too expensive)
    param_combinations = list(itertools.product(*param_grid.values()))
    np.random.shuffle(param_combinations)
    
    # Test top 6 combinations
    for i, params in enumerate(param_combinations[:6]):
        print(f"\nTesting combination {i+1}/6: {params}")
        
        # Create config with these parameters
        cfg = EnhancedConfig()
        cfg.__dict__.update(base_cfg.__dict__)
        cfg.nn_lr = params[0]
        cfg.nn_dropout = params[1]
        cfg.nn_batch_size = params[2]
        cfg.nn_hidden_layers = params[3]
        
        # Train single model for quick evaluation
        tf.random.set_seed(cfg.random_state)
        model = build_enhanced_meta_nn(num_dim, cat_dim, cfg)
        
        # Class weights
        class_weight = None
        if cfg.nn_use_class_weight_balanced:
            classes = np.unique(y_train)
            counts = np.bincount(y_train)
            total = counts.sum()
            class_weight = {cls: total / (len(classes) * counts[cls]) for cls in classes}
        
        # Quick training with fewer epochs
        callbacks = [EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=0)]
        
        history = model.fit(
            [X_train_num, X_train_cat], y_train.values,
            epochs=50,  # Reduced for speed
            batch_size=cfg.nn_batch_size,
            validation_split=cfg.nn_val_split,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=0
        )
        
        # Evaluate on validation set
        val_size = int(len(X_train_num) * cfg.nn_val_split)
        X_val_num = X_train_num[-val_size:]
        X_val_cat = X_train_cat[-val_size:]
        y_val = y_train.values[-val_size:]
        
        val_pred = model.predict([X_val_num, X_val_cat], verbose=0).ravel()
        val_auc = roc_auc_score(y_val, val_pred)
        
        print(f"Validation AUC: {val_auc:.4f}")
        
        if val_auc > best_score:
            best_score = val_auc
            best_params = params
            best_model = model
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best validation AUC: {best_score:.4f}")
    
    return best_params, best_model

# ------------------------------------------------------------
# Main enhanced training function
# ------------------------------------------------------------
def train_enhanced_meta_nn(X_train_num, X_train_cat, y_train, X_test_num, X_test_cat, 
                          num_dim, cat_dim, base_cfg: EnhancedConfig, optimize_hyperparams: bool = True):
    """Train enhanced meta neural network with all improvements"""
    
    if optimize_hyperparams:
        print("Step 1: Hyperparameter optimization...")
        best_params, _ = optimize_hyperparameters(X_train_num, X_train_cat, y_train, num_dim, cat_dim, base_cfg)
        
        # Update config with best parameters
        base_cfg.nn_lr = best_params[0]
        base_cfg.nn_dropout = best_params[1]
        base_cfg.nn_batch_size = best_params[2]
        base_cfg.nn_hidden_layers = best_params[3]
    
    print("\nStep 2: Training ensemble with best parameters...")
    models, _ = train_ensemble(X_train_num, X_train_cat, y_train, num_dim, cat_dim, base_cfg)
    
    print("\nStep 3: Making ensemble predictions...")
    ensemble_pred = ensemble_predict(models, X_test_num, X_test_cat, base_cfg)
    
    return ensemble_pred, models

# ------------------------------------------------------------
# Utility functions (reused from original)
# ------------------------------------------------------------
def load_data(cfg) -> pd.DataFrame:
    df = pd.read_csv(cfg.csv_path)
    for c in cfg.drop_cols:
        if c in df.columns:
            df = df.drop(columns=c)
    return df

def split_cols(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    y = df[target].astype(int)
    X = df.drop(columns=target)
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return X, y, num_cols, cat_cols

def make_numeric_preproc():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

def make_categorical_preproc():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

def fit_transform_preprocessors(X_train, X_test, num_cols, cat_cols):
    num_pre = make_numeric_preproc()
    cat_pre = make_categorical_preproc()
    X_train_num = num_pre.fit_transform(X_train[num_cols])
    X_test_num  = num_pre.transform(X_test[num_cols])
    X_train_cat = cat_pre.fit_transform(X_train[cat_cols]) if len(cat_cols) > 0 else np.empty((len(X_train), 0))
    X_test_cat  = cat_pre.transform(X_test[cat_cols]) if len(cat_cols) > 0 else np.empty((len(X_test), 0))
    return num_pre, cat_pre, X_train_num, X_test_num, X_train_cat, X_test_cat

# Business evaluation functions (reused from original)
def business_eval(y_true: np.ndarray, y_prob: np.ndarray, threshold: float,
                  revenue_per_good: float, loss_per_default: float) -> Dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    revenue = tn * revenue_per_good
    loss = fn * loss_per_default
    profit = revenue - loss
    return dict(
        threshold=threshold, tn=tn, fp=fp, fn=fn, tp=tp,
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        accuracy=accuracy_score(y_true, y_pred),
        revenue=revenue, loss=loss, profit=profit
    )

def sweep_thresholds(y_true: np.ndarray, y_prob: np.ndarray, low: float, high: float, points: int,
                     revenue_per_good: float, loss_per_default: float):
    thresholds = np.linspace(low, high, points)
    grid = [business_eval(y_true, y_prob, t, revenue_per_good, loss_per_default) for t in thresholds]
    best = max(grid, key=lambda r: r["profit"])
    return best, grid

def print_summary_block(name: str, prob, y_test, cfg):
    # Default 0.50
    y_pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, prob)

    print(f"\n{name} — Test @ threshold=0.50")
    print("-" * len(f"{name} — Test @ threshold=0.50"))
    print(f"Accuracy:  {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC AUC: {auc:.4f}")
    print(f"Confusion  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")

    best, grid = sweep_thresholds(y_test, prob, cfg.threshold_low, cfg.threshold_high, cfg.threshold_points,
                                  cfg.revenue_per_good, cfg.loss_per_default)
    print(f"\n{name} — Business-optimal threshold")
    print("-" * len(f"{name} — Business-optimal threshold"))
    print(f"Threshold: {best['threshold']:.4f} | Profit: ${best['profit']:,.0f}")
    print(f"Revenue:   ${best['revenue']:,.0f} | Loss: ${best['loss']:,.0f}")
    print(f"Accuracy:  {best['accuracy']:.4f} | Precision: {best['precision']:.4f} | Recall: {best['recall']:.4f} | F1: {best['f1']:.4f}")
    print(f"Confusion  TN={best['tn']:,}  FP={best['fp']:,}  FN={best['fn']:,}  TP={best['tp']:,}")
    return best

if __name__ == "__main__":
    # Test the enhanced model
    cfg = EnhancedConfig()
    
    print("Loading data...")
    df = load_data(cfg)
    X, y, num_cols, cat_cols = split_cols(df, cfg.target)
    print(f"Dataset: {df.shape} | Default rate={y.mean():.3%}")
    print(f"Numeric: {len(num_cols)} | Categorical: {len(cat_cols)}")
    
    # Same split as original
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )
    
    # Preprocessing
    num_pre, cat_pre, Xtr_num, Xte_num, Xtr_cat, Xte_cat = fit_transform_preprocessors(
        X_train, X_test, num_cols, cat_cols
    )
    
    # Train enhanced model
    enhanced_pred, models = train_enhanced_meta_nn(
        Xtr_num, Xtr_cat, y_train, Xte_num, Xte_cat, 
        Xtr_num.shape[1], Xtr_cat.shape[1], cfg, optimize_hyperparams=True
    )
    
    # Evaluate
    best_enhanced = print_summary_block("Enhanced Meta NN", enhanced_pred, y_test.values, cfg)
    print(f"\nEnhanced Meta NN achieved ${best_enhanced['profit']:,.0f} profit")

