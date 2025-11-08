"""
Model Implementations for EMBS Dataset
=======================================

This module contains model implementations for loan default prediction:
1. LogisticRegressionModel: Baseline linear model
2. SimpleMetaNN: Basic two-branch meta neural network
3. EnhancedMetaNN: Advanced meta neural network with attention and residual connections
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Concatenate, BatchNormalization, 
    Multiply, Add
)
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


class LogisticRegressionModel:
    """
    Logistic Regression baseline model
    Combines origination and performance features
    """
    
    def __init__(self, random_state=42, max_iter=1000):
        self.random_state = random_state
        self.max_iter = max_iter
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            solver='lbfgs',
            max_iter=max_iter,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
        
    def fit(self, X_orig, X_perf, y):
        """
        Fit the logistic regression model
        
        Args:
            X_orig: Origination features
            X_perf: Performance features
            y: Target variable
        """
        # Combine origination and performance features
        X_combined = np.hstack([X_orig, X_perf])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_combined)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        return self
    
    def predict_proba(self, X_orig, X_perf):
        """
        Predict probability of default
        
        Args:
            X_orig: Origination features
            X_perf: Performance features
            
        Returns:
            Array of default probabilities
        """
        # Combine and scale features
        X_combined = np.hstack([X_orig, X_perf])
        X_scaled = self.scaler.transform(X_combined)
        
        # Get probabilities for positive class (default)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def predict(self, X_orig, X_perf, threshold=0.5):
        """Predict binary default status"""
        proba = self.predict_proba(X_orig, X_perf)
        return (proba >= threshold).astype(int)


class SimpleMetaNN:
    """
    Simple Meta Neural Network with two branches:
    - Origination branch: processes static loan features
    - Performance branch: processes dynamic performance features
    """
    
    def __init__(self, orig_dim, perf_dim, random_state=42):
        self.orig_dim = orig_dim
        self.perf_dim = perf_dim
        self.random_state = random_state
        self.model = None
        self._build_model()
        
    def _build_model(self):
        """Build the two-branch meta neural network"""
        # Origination branch
        orig_input = Input(shape=(self.orig_dim,), name="orig_input")
        x_orig = Dense(64, activation="relu")(orig_input)
        x_orig = Dropout(0.3)(x_orig)
        x_orig = Dense(32, activation="relu")(x_orig)
        
        # Performance branch
        perf_input = Input(shape=(self.perf_dim,), name="perf_input")
        x_perf = Dense(64, activation="relu")(perf_input)
        x_perf = Dropout(0.3)(x_perf)
        x_perf = Dense(32, activation="relu")(x_perf)
        
        # Fusion layer
        fused = Concatenate(name="fusion")([x_orig, x_perf])
        z = Dense(32, activation="relu")(fused)
        z = Dropout(0.3)(z)
        z = Dense(16, activation="relu")(z)
        
        # Output layer
        output = Dense(1, activation="sigmoid", name="default_prob")(z)
        
        # Create and compile model
        self.model = Model(
            inputs=[orig_input, perf_input], 
            outputs=output, 
            name="simple_meta_nn"
        )
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
    def fit(self, X_orig, X_perf, y, epochs=100, batch_size=32, verbose=0):
        """
        Train the meta neural network
        
        Args:
            X_orig: Origination features
            X_perf: Performance features
            y: Target variable
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
        """
        # Use early stopping
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=10,
            restore_best_weights=True
        )
        
        self.model.fit(
            [X_orig, X_perf],
            y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=[early_stop],
            validation_split=0.1
        )
        
        return self
    
    def predict_proba(self, X_orig, X_perf):
        """
        Predict probability of default
        
        Args:
            X_orig: Origination features
            X_perf: Performance features
            
        Returns:
            Array of default probabilities
        """
        return self.model.predict([X_orig, X_perf], verbose=0).flatten()
    
    def predict(self, X_orig, X_perf, threshold=0.5):
        """Predict binary default status"""
        proba = self.predict_proba(X_orig, X_perf)
        return (proba >= threshold).astype(int)


class EnhancedMetaNN:
    """
    Enhanced Meta Neural Network with advanced features:
    - Batch normalization
    - Residual connections
    - Attention mechanism
    - Deeper architecture
    """
    
    def __init__(self, orig_dim, perf_dim, random_state=42):
        self.orig_dim = orig_dim
        self.perf_dim = perf_dim
        self.random_state = random_state
        self.model = None
        self._build_model()
        
    def _residual_block(self, x, units, dropout_rate=0.3):
        """Residual block with skip connection"""
        # Main path
        out = Dense(units, activation="relu")(x)
        out = BatchNormalization()(out)
        out = Dropout(dropout_rate)(out)
        
        # Skip connection
        if x.shape[-1] != units:
            x = Dense(units)(x)  # Project to same dimension
        
        # Add skip connection
        out = Add()([out, x])
        return out
    
    def _attention_layer(self, x, name="attention"):
        """Attention mechanism"""
        # Calculate attention weights
        attention = Dense(x.shape[-1], activation="tanh")(x)
        attention_weights = Dense(x.shape[-1], activation="softmax")(attention)
        
        # Apply attention
        attended = Multiply(name=f"{name}_out")([x, attention_weights])
        return attended
        
    def _build_model(self):
        """Build the enhanced meta neural network"""
        # Origination branch with residual blocks
        orig_input = Input(shape=(self.orig_dim,), name="orig_input")
        x_orig = Dense(128, activation="relu")(orig_input)
        x_orig = BatchNormalization()(x_orig)
        x_orig = Dropout(0.3)(x_orig)
        x_orig = self._residual_block(x_orig, 64)
        x_orig = Dense(32, activation="relu")(x_orig)
        
        # Performance branch with residual blocks
        perf_input = Input(shape=(self.perf_dim,), name="perf_input")
        x_perf = Dense(128, activation="relu")(perf_input)
        x_perf = BatchNormalization()(x_perf)
        x_perf = Dropout(0.3)(x_perf)
        x_perf = self._residual_block(x_perf, 64)
        x_perf = Dense(32, activation="relu")(x_perf)
        
        # Fusion layer with attention
        fused = Concatenate(name="fusion")([x_orig, x_perf])
        fused = self._attention_layer(fused, name="attention")
        
        # Deep layers
        z = Dense(64, activation="relu")(fused)
        z = BatchNormalization()(z)
        z = Dropout(0.3)(z)
        z = Dense(32, activation="relu")(z)
        z = BatchNormalization()(z)
        z = Dropout(0.2)(z)
        z = Dense(16, activation="relu")(z)
        
        # Output layer
        output = Dense(1, activation="sigmoid", name="default_prob")(z)
        
        # Create and compile model
        self.model = Model(
            inputs=[orig_input, perf_input], 
            outputs=output, 
            name="enhanced_meta_nn"
        )
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
    def fit(self, X_orig, X_perf, y, epochs=100, batch_size=32, verbose=0):
        """
        Train the enhanced meta neural network
        
        Args:
            X_orig: Origination features
            X_perf: Performance features
            y: Target variable
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
        """
        # Callbacks for training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6
            )
        ]
        
        self.model.fit(
            [X_orig, X_perf],
            y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
            validation_split=0.1
        )
        
        return self
    
    def predict_proba(self, X_orig, X_perf):
        """
        Predict probability of default
        
        Args:
            X_orig: Origination features
            X_perf: Performance features
            
        Returns:
            Array of default probabilities
        """
        return self.model.predict([X_orig, X_perf], verbose=0).flatten()
    
    def predict(self, X_orig, X_perf, threshold=0.5):
        """Predict binary default status"""
        proba = self.predict_proba(X_orig, X_perf)
        return (proba >= threshold).astype(int)


# Model summary function for testing
def print_model_summaries():
    """Print summaries of all models (for testing)"""
    print("=" * 60)
    print("MODEL ARCHITECTURES")
    print("=" * 60)
    
    # Simple Meta NN
    print("\n1. SIMPLE META NEURAL NETWORK")
    print("-" * 60)
    simple_model = SimpleMetaNN(orig_dim=10, perf_dim=10)
    simple_model.model.summary()
    
    # Enhanced Meta NN
    print("\n2. ENHANCED META NEURAL NETWORK")
    print("-" * 60)
    enhanced_model = EnhancedMetaNN(orig_dim=10, perf_dim=10)
    enhanced_model.model.summary()


if __name__ == "__main__":
    print("✅ Models module loaded successfully!")
    print("\nAvailable models:")
    print("  - LogisticRegressionModel: Baseline linear model")
    print("  - SimpleMetaNN: Two-branch meta neural network")
    print("  - EnhancedMetaNN: Advanced meta neural network")
    print("\nRun print_model_summaries() to see model architectures")


