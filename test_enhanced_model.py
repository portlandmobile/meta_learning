# test_enhanced_model.py
# ------------------------------------------------------------
# Quick test script to validate enhanced meta NN model
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import tensorflow as tf
from improved_meta_nn import EnhancedConfig, build_enhanced_meta_nn, focal_loss

def test_model_creation():
    """Test that enhanced model can be created and compiled"""
    print("Testing enhanced model creation...")
    
    cfg = EnhancedConfig()
    cfg.nn_epochs = 5  # Quick test
    cfg.nn_n_models = 1  # Single model for test
    
    # Create model
    num_dim, cat_dim = 10, 5
    model = build_enhanced_meta_nn(num_dim, cat_dim, cfg, model_id=0)
    
    print(f"✓ Model created successfully")
    print(f"✓ Model name: {model.name}")
    print(f"✓ Input shapes: {[inp.shape for inp in model.inputs]}")
    print(f"✓ Output shape: {model.output.shape}")
    print(f"✓ Total parameters: {model.count_params():,}")
    
    # Test forward pass
    X_num = np.random.randn(32, num_dim).astype(np.float32)
    X_cat = np.random.randn(32, cat_dim).astype(np.float32)
    
    prediction = model([X_num, X_cat])
    print(f"✓ Forward pass successful: output shape {prediction.shape}")
    
    return model

def test_focal_loss():
    """Test focal loss implementation"""
    print("\nTesting focal loss...")
    
    # Create test data
    y_true = tf.constant([[0.0], [1.0], [0.0], [1.0]])
    y_pred = tf.constant([[0.1], [0.8], [0.3], [0.7]])
    
    # Test focal loss
    focal_loss_fn = focal_loss(alpha=0.25, gamma=2.0)
    loss_value = focal_loss_fn(y_true, y_pred)
    
    print(f"✓ Focal loss computed: {loss_value:.4f}")
    
    # Test binary crossentropy for comparison
    bce = tf.keras.losses.BinaryCrossentropy()
    bce_value = bce(y_true, y_pred)
    print(f"✓ Binary crossentropy for comparison: {bce_value:.4f}")
    
    return True

def test_config():
    """Test enhanced configuration"""
    print("\nTesting enhanced configuration...")
    
    cfg = EnhancedConfig()
    
    # Check key parameters
    assert cfg.nn_use_class_weight_balanced == True
    assert cfg.nn_use_focal_loss == True
    assert cfg.nn_use_batch_norm == True
    assert cfg.nn_use_residual == True
    assert cfg.nn_use_attention == True
    
    print("✓ Enhanced configuration validated")
    print(f"✓ Class weighting enabled: {cfg.nn_use_class_weight_balanced}")
    print(f"✓ Focal loss enabled: {cfg.nn_use_focal_loss}")
    print(f"✓ Batch normalization enabled: {cfg.nn_use_batch_norm}")
    print(f"✓ Residual connections enabled: {cfg.nn_use_residual}")
    print(f"✓ Attention mechanism enabled: {cfg.nn_use_attention}")
    print(f"✓ Hidden layers: {cfg.nn_hidden_layers}")
    
    return cfg

if __name__ == "__main__":
    print("="*60)
    print("TESTING ENHANCED META NN MODEL")
    print("="*60)
    
    # Set random seed
    tf.random.set_seed(42)
    np.random.seed(42)
    
    try:
        # Test configuration
        cfg = test_config()
        
        # Test focal loss
        test_focal_loss()
        
        # Test model creation
        model = test_model_creation()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print("The enhanced meta NN model is ready for training!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

