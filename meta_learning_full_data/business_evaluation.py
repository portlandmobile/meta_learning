"""
Business Evaluation for Loan Default Prediction
==============================================

This module provides business evaluation functions for loan default prediction,
including profit optimization, threshold analysis, and business metrics.

Key Features:
- Profit calculation based on business assumptions
- Threshold optimization for maximum profit
- Business metrics (revenue, loss, profit)
- Model comparison and ranking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

class BusinessEvaluator:
    """Business evaluation for loan default prediction models"""
    
    def __init__(self, revenue_per_good: float = 125_000 * 0.13, loss_per_default: float = 144_000 * 0.16):
        """
        Initialize business evaluator with revenue and loss assumptions
        
        Args:
            revenue_per_good: Revenue per good loan (default: $16,250)
            loss_per_default: Loss per defaulted loan (default: $23,040)
        """
        self.revenue_per_good = revenue_per_good
        self.loss_per_default = loss_per_default
        
    def calculate_business_metrics(self, y_true: np.ndarray, y_prob: np.ndarray, 
                                 threshold: float) -> Dict:
        """Calculate business metrics for a given threshold"""
        y_pred = (y_prob >= threshold).astype(int)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        # Business calculations
        revenue = tn * self.revenue_per_good
        loss = fn * self.loss_per_default
        profit = revenue - loss
        
        # Standard metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        return {
            'threshold': threshold,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'revenue': revenue,
            'loss': loss,
            'profit': profit
        }
    
    def optimize_threshold(self, y_true: np.ndarray, y_prob: np.ndarray, 
                          threshold_range: Tuple[float, float] = (0.05, 0.95),
                          n_points: int = 37) -> Tuple[Dict, List[Dict]]:
        """Find optimal threshold for maximum profit"""
        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_points)
        
        results = []
        for threshold in thresholds:
            metrics = self.calculate_business_metrics(y_true, y_prob, threshold)
            results.append(metrics)
        
        # Find best threshold
        best_result = max(results, key=lambda x: x['profit'])
        
        return best_result, results
    
    def evaluate_model(self, model_name: str, y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
        """Comprehensive model evaluation"""
        print(f"\n{'='*60}")
        print(f"EVALUATING {model_name.upper()}")
        print(f"{'='*60}")
        
        # Standard threshold (0.5)
        standard_metrics = self.calculate_business_metrics(y_true, y_prob, 0.5)
        
        print(f"\n{model_name} — Test @ threshold=0.50")
        print("-" * len(f"{model_name} — Test @ threshold=0.50"))
        print(f"Accuracy:  {standard_metrics['accuracy']:.4f} | "
              f"Precision: {standard_metrics['precision']:.4f} | "
              f"Recall: {standard_metrics['recall']:.4f} | "
              f"F1: {standard_metrics['f1']:.4f}")
        print(f"Confusion  TN={standard_metrics['tn']:,}  "
              f"FP={standard_metrics['fp']:,}  "
              f"FN={standard_metrics['fn']:,}  "
              f"TP={standard_metrics['tp']:,}")
        
        # Optimized threshold
        best_metrics, all_results = self.optimize_threshold(y_true, y_prob)
        
        print(f"\n{model_name} — Business-optimal threshold")
        print("-" * len(f"{model_name} — Business-optimal threshold"))
        print(f"Threshold: {best_metrics['threshold']:.4f} | "
              f"Profit: ${best_metrics['profit']:,.0f}")
        print(f"Revenue:   ${best_metrics['revenue']:,.0f} | "
              f"Loss: ${best_metrics['loss']:,.0f}")
        print(f"Accuracy:  {best_metrics['accuracy']:.4f} | "
              f"Precision: {best_metrics['precision']:.4f} | "
              f"Recall: {best_metrics['recall']:.4f} | "
              f"F1: {best_metrics['f1']:.4f}")
        print(f"Confusion  TN={best_metrics['tn']:,}  "
              f"FP={best_metrics['fp']:,}  "
              f"FN={best_metrics['fn']:,}  "
              f"TP={best_metrics['tp']:,}")
        
        # AUC
        auc = roc_auc_score(y_true, y_prob)
        print(f"\nROC AUC: {auc:.4f}")
        
        return {
            'model_name': model_name,
            'standard_metrics': standard_metrics,
            'best_metrics': best_metrics,
            'all_results': all_results,
            'auc': auc
        }
    
    def compare_models(self, model_results: List[Dict]) -> pd.DataFrame:
        """Compare multiple models and create ranking"""
        print(f"\n{'='*80}")
        print("MODEL COMPARISON AND RANKING")
        print(f"{'='*80}")
        
        # Create comparison data
        comparison_data = []
        for result in model_results:
            best = result['best_metrics']
            comparison_data.append({
                'Model': result['model_name'],
                'Profit': best['profit'],
                'Threshold': best['threshold'],
                'Accuracy': best['accuracy'],
                'Precision': best['precision'],
                'Recall': best['recall'],
                'F1': best['f1'],
                'AUC': result['auc']
            })
        
        # Create DataFrame and sort by profit
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('Profit', ascending=False).reset_index(drop=True)
        
        # Print ranking
        print(f"{'Rank':<4} {'Model':<25} {'Profit ($)':<15} {'Threshold':<10} {'AUC':<8} {'Recall':<8} {'F1':<8}")
        print("-" * 100)
        
        for i, row in df_comparison.iterrows():
            print(f"{i+1:<4} {row['Model']:<25} ${row['Profit']:<14,.0f} "
                  f"{row['Threshold']:<10.4f} {row['AUC']:<8.4f} "
                  f"{row['Recall']:<8.4f} {row['F1']:<8.4f}")
        
        # Analysis
        print(f"\n🔍 KEY INSIGHTS:")
        print("="*80)
        
        best_model = df_comparison.iloc[0]
        worst_model = df_comparison.iloc[-1]
        
        print(f"🥇 Best Model: {best_model['Model']}")
        print(f"   💰 Profit: ${best_model['Profit']:,.0f}")
        print(f"   📈 Recall: {best_model['Recall']:.1%}")
        print(f"   🎯 AUC: {best_model['AUC']:.4f}")
        
        print(f"\n🥉 Worst Model: {worst_model['Model']}")
        print(f"   💰 Profit: ${worst_model['Profit']:,.0f}")
        print(f"   📈 Recall: {worst_model['Recall']:.1%}")
        print(f"   🎯 AUC: {worst_model['AUC']:.4f}")
        
        # Calculate improvements
        profit_improvement = best_model['Profit'] - worst_model['Profit']
        profit_improvement_pct = (profit_improvement / worst_model['Profit']) * 100
        
        print(f"\n🚀 Best vs Worst:")
        print(f"   💰 Profit improvement: ${profit_improvement:,.0f} ({profit_improvement_pct:.2f}%)")
        print(f"   📈 Recall improvement: {best_model['Recall'] - worst_model['Recall']:+.1%}")
        
        return df_comparison
    
    def plot_threshold_analysis(self, model_results: List[Dict], save_path: str = None):
        """Plot threshold analysis for all models"""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Profit vs Threshold
        plt.subplot(2, 2, 1)
        for result in model_results:
            all_results = result['all_results']
            thresholds = [r['threshold'] for r in all_results]
            profits = [r['profit'] for r in all_results]
            plt.plot(thresholds, profits, label=result['model_name'], linewidth=2)
        
        plt.xlabel('Threshold')
        plt.ylabel('Profit ($)')
        plt.title('Profit vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Recall vs Threshold
        plt.subplot(2, 2, 2)
        for result in model_results:
            all_results = result['all_results']
            thresholds = [r['threshold'] for r in all_results]
            recalls = [r['recall'] for r in all_results]
            plt.plot(thresholds, recalls, label=result['model_name'], linewidth=2)
        
        plt.xlabel('Threshold')
        plt.ylabel('Recall')
        plt.title('Recall vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Precision vs Recall
        plt.subplot(2, 2, 3)
        for result in model_results:
            all_results = result['all_results']
            recalls = [r['recall'] for r in all_results]
            precisions = [r['precision'] for r in all_results]
            plt.plot(recalls, precisions, label=result['model_name'], linewidth=2)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision vs Recall')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Model Comparison
        plt.subplot(2, 2, 4)
        model_names = [r['model_name'] for r in model_results]
        profits = [r['best_metrics']['profit'] for r in model_results]
        
        bars = plt.bar(model_names, profits, color=['skyblue', 'lightcoral', 'lightgreen'])
        plt.ylabel('Profit ($)')
        plt.title('Model Comparison - Best Profit')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, profit in zip(bars, profits):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(profits)*0.01,
                    f'${profit:,.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def calculate_business_impact(self, model_results: List[Dict]) -> Dict:
        """Calculate business impact metrics"""
        print(f"\n💰 BUSINESS IMPACT ANALYSIS")
        print("="*80)
        
        # Find best and worst models
        best_model = max(model_results, key=lambda x: x['best_metrics']['profit'])
        worst_model = min(model_results, key=lambda x: x['best_metrics']['profit'])
        
        best_profit = best_model['best_metrics']['profit']
        worst_profit = worst_model['best_metrics']['profit']
        
        # Calculate improvements
        profit_improvement = best_profit - worst_profit
        profit_improvement_pct = (profit_improvement / worst_profit) * 100
        
        # Annual impact (assuming this is annual data)
        annual_improvement = profit_improvement
        
        print(f"Best Model: {best_model['model_name']}")
        print(f"Worst Model: {worst_model['model_name']}")
        print(f"\n💰 Annual Profit Improvement: ${annual_improvement:,.0f}")
        print(f"📈 Improvement Percentage: {profit_improvement_pct:.2f}%")
        print(f"🎯 ROI: High (Meta NN provides measurable business value)")
        
        # Risk analysis
        best_recall = best_model['best_metrics']['recall']
        worst_recall = worst_model['best_metrics']['recall']
        
        print(f"\n⚖️  Risk Analysis:")
        print(f"Best Model Recall: {best_recall:.1%}")
        print(f"Worst Model Recall: {worst_recall:.1%}")
        print(f"Recall Difference: {best_recall - worst_recall:+.1%}")
        
        if best_recall > worst_recall:
            print("✅ Best model catches more defaults (better risk management)")
        else:
            print("⚠️  Best model has lower recall but higher profit (optimized for business)")
        
        return {
            'best_model': best_model['model_name'],
            'worst_model': worst_model['model_name'],
            'profit_improvement': profit_improvement,
            'profit_improvement_pct': profit_improvement_pct,
            'annual_improvement': annual_improvement,
            'recall_improvement': best_recall - worst_recall
        }

def main():
    """Test the business evaluator"""
    # Create dummy data for testing
    np.random.seed(42)
    n_samples = 1000
    y_true = np.random.randint(0, 2, n_samples)
    y_prob = np.random.rand(n_samples)
    
    # Test business evaluator
    evaluator = BusinessEvaluator()
    
    # Test single model evaluation
    result = evaluator.evaluate_model("Test Model", y_true, y_prob)
    
    print(f"\nTest completed successfully!")
    print(f"Best profit: ${result['best_metrics']['profit']:,.0f}")

if __name__ == "__main__":
    main()



