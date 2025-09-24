#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


def plot_multi_model_comparison(results, y_test):
    """
    Plot comprehensive comparison of multiple models
    
    Args:
        results: Dictionary from train_advanced_meta_learning_models()
        y_test: Test labels for metrics calculation
    """
    n_models = len(results)
    model_names = list(results.keys())
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    fig.suptitle('Multi-Model Meta-Learning Comparison', fontsize=16, fontweight='bold')
    
    # Colors for different models
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown'][:n_models]
    
    # 1. Training History - Accuracy
    ax = axes[0, 0]
    for i, (name, result) in enumerate(results.items()):
        history = result['history']
        ax.plot(history.history['accuracy'], label=f'{name} Train', 
                color=colors[i], linestyle='-', alpha=0.8)
        ax.plot(history.history['val_accuracy'], label=f'{name} Val', 
                color=colors[i], linestyle='--', alpha=0.8)
    ax.set_title('Training Accuracy Comparison')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    
    # 2. Training History - Loss
    ax = axes[0, 1]
    for i, (name, result) in enumerate(results.items()):
        history = result['history']
        ax.plot(history.history['loss'], label=f'{name} Train', 
                color=colors[i], linestyle='-', alpha=0.8)
        ax.plot(history.history['val_loss'], label=f'{name} Val', 
                color=colors[i], linestyle='--', alpha=0.8)
    ax.set_title('Training Loss Comparison')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    
    # 3. Prediction Distribution
    ax = axes[0, 2]
    for i, (name, result) in enumerate(results.items()):
        predictions = result['predictions']['probabilities'].flatten()
        ax.hist(predictions, alpha=0.6, label=name, bins=30, 
                color=colors[i], density=True)
    ax.set_title('Prediction Probability Distribution')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True)
    
    # 4. Business Performance Comparison
    ax = axes[0, 3]
    profits = [result['profit'] for result in results.values()]
    thresholds = [result['threshold']['threshold'] for result in results.values()]
    
    bars = ax.bar(model_names, profits, color=colors, alpha=0.7)
    ax.set_title('Business Performance (Profit)')
    ax.set_ylabel('Profit ($)')
    ax.tick_params(axis='x', rotation=45)
    
    # Add profit values on bars
    for bar, profit in zip(bars, profits):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'${profit:,.0f}', ha='center', va='bottom', fontsize=10)
    ax.grid(True, axis='y')
    
    # 5. Threshold Comparison
    ax = axes[1, 0]
    bars = ax.bar(model_names, thresholds, color=colors, alpha=0.7)
    ax.set_title('Optimal Thresholds')
    ax.set_ylabel('Threshold')
    ax.tick_params(axis='x', rotation=45)
    
    # Add threshold values on bars
    for bar, thresh in zip(bars, thresholds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{thresh:.3f}', ha='center', va='bottom', fontsize=10)
    ax.grid(True, axis='y')
    
    # 6. Precision-Recall Comparison
    ax = axes[1, 1]
    for i, (name, result) in enumerate(results.items()):
        thresh_info = result['threshold']
        ax.scatter(thresh_info['recall'], thresh_info['precision'], 
                  s=100, color=colors[i], label=name, alpha=0.8)
        ax.annotate(name, (thresh_info['recall'], thresh_info['precision']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax.set_title('Precision vs Recall at Optimal Thresholds')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend()
    ax.grid(True)
    
    # 7. Confusion Matrix Heatmaps (for best model)
    best_model_name = max(results.keys(), key=lambda x: results[x]['profit'])
    best_result = results[best_model_name]
    
    cm = confusion_matrix(y_test, best_result['predictions']['binary'])
    ax = axes[1, 2]
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f'Confusion Matrix - {best_model_name}\n(Best Performer)')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   horizontalalignment="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No Default', 'Default'])
    ax.set_yticklabels(['No Default', 'Default'])
    
    # 8. Model Complexity Comparison (parameter count)
    ax = axes[1, 3]
    param_counts = []
    for name, result in results.items():
        param_count = result['model'].count_params()
        param_counts.append(param_count)
    
    bars = ax.bar(model_names, param_counts, color=colors, alpha=0.7)
    ax.set_title('Model Complexity (Parameters)')
    ax.set_ylabel('Number of Parameters')
    ax.tick_params(axis='x', rotation=45)
    
    # Add parameter counts on bars
    for bar, count in zip(bars, param_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{count:,}', ha='center', va='bottom', fontsize=9)
    ax.grid(True, axis='y')
    
    # 9. Performance Metrics Radar Chart
    from math import pi
    
    ax = axes[2, 0]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
    angles += angles[:1]  # Complete the circle
    
    ax = plt.subplot(3, 4, 9, projection='polar')
    
    for i, (name, result) in enumerate(results.items()):
        thresh_info = result['threshold']
        values = [
            thresh_info['accuracy'],
            thresh_info['precision'], 
            thresh_info['recall'],
            thresh_info['f1']
        ]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title('Performance Metrics Comparison', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # 10. Training Convergence Speed
    ax = axes[2, 1]
    for i, (name, result) in enumerate(results.items()):
        history = result['history']
        # Find epoch where validation loss stops improving significantly
        val_losses = history.history['val_loss']
        convergence_epoch = len(val_losses)
        for epoch in range(5, len(val_losses)):
            if all(val_losses[epoch] >= val_losses[epoch-j] for j in range(1, 6)):
                convergence_epoch = epoch
                break
        
        ax.bar(name, convergence_epoch, color=colors[i], alpha=0.7)
    
    ax.set_title('Training Convergence Speed')
    ax.set_ylabel('Epochs to Convergence')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, axis='y')
    
    # 11. ROC Comparison
    ax = axes[2, 2]
    from sklearn.metrics import roc_curve, auc
    
    for i, (name, result) in enumerate(results.items()):
        y_prob = result['predictions']['probabilities'].flatten()
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=2, 
                label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison')
    ax.legend(loc="lower right")
    ax.grid(True)
    
    # 12. Business Impact Summary
    ax = axes[2, 3]
    
    # Calculate business metrics for each model
    metrics_data = []
    for name, result in results.items():
        thresh_info = result['threshold']
        metrics_data.append({
            'Model': name,
            'Profit': result['profit'],
            'TP': thresh_info['tp'],
            'TN': thresh_info['tn'], 
            'FP': thresh_info['fp'],
            'FN': thresh_info['fn']
        })
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Create stacked bar chart of TP, TN, FP, FN
    bottoms_pos = np.zeros(len(df_metrics))
    bottoms_neg = np.zeros(len(df_metrics))
    
    ax.bar(df_metrics['Model'], df_metrics['TP'], label='True Positives', 
           color='darkgreen', alpha=0.8)
    ax.bar(df_metrics['Model'], df_metrics['TN'], bottom=df_metrics['TP'],
           label='True Negatives', color='lightgreen', alpha=0.8)
    
    bottoms = df_metrics['TP'] + df_metrics['TN']
    ax.bar(df_metrics['Model'], df_metrics['FP'], bottom=bottoms,
           label='False Positives', color='orange', alpha=0.8)
    
    bottoms += df_metrics['FP'] 
    ax.bar(df_metrics['Model'], df_metrics['FN'], bottom=bottoms,
           label='False Negatives', color='red', alpha=0.8)
    
    ax.set_title('Prediction Breakdown by Model')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    plt.show()

def print_detailed_model_comparison(results, y_test):
    """
    Print detailed comparison of all models
    """
    print("="*100)
    print("DETAILED MODEL COMPARISON REPORT")
    print("="*100)
    
    # Sort models by profit
    sorted_results = sorted(results.items(), key=lambda x: x[1]['profit'], reverse=True)
    
    print(f"\n{'RANKING BY BUSINESS PERFORMANCE:':<50}")
    print("-"*100)
    for rank, (name, result) in enumerate(sorted_results, 1):
        profit = result['profit']
        threshold = result['threshold']['threshold']
        print(f"{rank}. {name:<20} | Profit: ${profit:>12,.0f} | Threshold: {threshold:.4f}")
    
    print(f"\n{'DETAILED METRICS COMPARISON:':<50}")
    print("-"*100)
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<12} {'Recall':<10} {'F1-Score':<10} {'Parameters':<12}")
    print("-"*100)
    
    for name, result in sorted_results:
        thresh_info = result['threshold']
        param_count = result['model'].count_params()
        print(f"{name:<20} {thresh_info['accuracy']:<10.4f} {thresh_info['precision']:<12.4f} "
              f"{thresh_info['recall']:<10.4f} {thresh_info['f1']:<10.4f} {param_count:<12,}")
    
    print(f"\n{'CONFUSION MATRIX BREAKDOWN:':<50}")
    print("-"*100)
    print(f"{'Model':<20} {'TP':<8} {'TN':<8} {'FP':<8} {'FN':<8} {'Precision':<12} {'Recall':<10}")
    print("-"*100)
    
    for name, result in sorted_results:
        thresh_info = result['threshold']
        print(f"{name:<20} {thresh_info['tp']:<8} {thresh_info['tn']:<8} "
              f"{thresh_info['fp']:<8} {thresh_info['fn']:<8} "
              f"{thresh_info['precision']:<12.4f} {thresh_info['recall']:<10.4f}")
    
    # Best model detailed report
    best_name, best_result = sorted_results[0]
    print(f"\n{'BEST MODEL DETAILED ANALYSIS:':<50}")
    print("="*100)
    print(f"Model: {best_name}")
    print(f"Profit: ${best_result['profit']:,.0f}")
    print(f"Optimal Threshold: {best_result['threshold']['threshold']:.4f}")
    print(f"Parameters: {best_result['model'].count_params():,}")
    
    print(f"\nClassification Report:")
    print("-"*50)
    print(classification_report(y_test, best_result['predictions']['binary'], 
                              target_names=['No Default', 'Default']))
    
    # Model architecture comparison
    print(f"\n{'MODEL ARCHITECTURE SUMMARY:':<50}")
    print("-"*100)
    for name, result in results.items():
        model = result['model']
        print(f"\n{name.upper()}:")
        print(f"  Total Parameters: {model.count_params():,}")
        print(f"  Trainable Parameters: {sum([np.prod(v.shape) for v in model.trainable_variables]):,}")
        print(f"  Layers: {len(model.layers)}")
        
        # Get input shapes
        if hasattr(model, 'input_shape'):
            if isinstance(model.input_shape, list):
                print(f"  Input Shapes: {[shape for shape in model.input_shape]}")
            else:
                print(f"  Input Shape: {model.input_shape}")

def analyze_model_performance_differences(results):
    """
    Analyze why different models perform differently
    """
    print(f"\n{'PERFORMANCE ANALYSIS:':<50}")
    print("="*100)
    
    # Find best and worst models
    sorted_by_profit = sorted(results.items(), key=lambda x: x[1]['profit'], reverse=True)
    best_name, best_result = sorted_by_profit[0]
    worst_name, worst_result = sorted_by_profit[-1]
    
    profit_diff = best_result['profit'] - worst_result['profit']
    threshold_diff = best_result['threshold']['threshold'] - worst_result['threshold']['threshold']
    
    print(f"Best Model: {best_name}")
    print(f"Worst Model: {worst_name}")
    print(f"Profit Difference: ${profit_diff:,.0f}")
    print(f"Threshold Difference: {threshold_diff:.4f}")
    
    # Analyze prediction differences
    best_pred = best_result['predictions']['probabilities'].flatten()
    worst_pred = worst_result['predictions']['probabilities'].flatten()
    
    pred_corr = np.corrcoef(best_pred, worst_pred)[0, 1]
    print(f"Prediction Correlation: {pred_corr:.4f}")
    
    print(f"\nPrediction Statistics:")
    print(f"  {best_name:<20} Mean: {np.mean(best_pred):.4f}, Std: {np.std(best_pred):.4f}")
    print(f"  {worst_name:<20} Mean: {np.mean(worst_pred):.4f}, Std: {np.std(worst_pred):.4f}")
    
    # Training efficiency
    best_epochs = len(best_result['history'].history['loss'])
    worst_epochs = len(worst_result['history'].history['loss'])
    
    print(f"\nTraining Efficiency:")
    print(f"  {best_name}: {best_epochs} epochs")
    print(f"  {worst_name}: {worst_epochs} epochs")
    
    return {
        'profit_difference': profit_diff,
        'prediction_correlation': pred_corr,
        'best_model': best_name,
        'worst_model': worst_name
    }


# In[ ]:


# Usage example:
def comprehensive_model_analysis(results, y_test):
    """
    Run complete analysis of all models
    """
    # Plot comprehensive comparison
    plot_multi_model_comparison(results, y_test)
    
    # Print detailed metrics
    print_detailed_model_comparison(results, y_test)
    
    # Analyze performance differences  
    analysis = analyze_model_performance_differences(results)
    
    return analysis

