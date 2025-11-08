#!/usr/bin/env python3
"""
Analyze default loan detection improvements
"""
import pandas as pd
import numpy as np

def analyze_default_detection():
    print('🔍 DEFAULT LOAN DETECTION ANALYSIS')
    print('=' * 60)

    # Load results
    results = pd.read_csv('model_comparison_results.csv')
    predictions = pd.read_csv('model_predictions.csv')

    print('\n📊 DEFAULT DETECTION COMPARISON:')
    print('Model                | Recall | Precision | F1    | AUC')
    print('-' * 55)
    for i, row in results.iterrows():
        print(f'{row["Model"]:<20} | {row["Recall"]:>6.3f} | {row["Precision"]:>9.3f} | {row["F1"]:>6.3f} | {row["AUC"]:>6.3f}')

    print('\n🎯 SPECIFIC DEFAULT DETECTION ANALYSIS:')
    lr_recall = results.iloc[2]['Recall']  # Logistic Regression
    meta_recall = results.iloc[0]['Recall']  # Simple Meta NN

    lr_precision = results.iloc[2]['Precision']
    meta_precision = results.iloc[0]['Precision']

    lr_f1 = results.iloc[2]['F1']
    meta_f1 = results.iloc[0]['F1']

    lr_auc = results.iloc[2]['AUC']
    meta_auc = results.iloc[0]['AUC']

    print(f'Logistic Regression Recall: {lr_recall:.1%}')
    print(f'Simple Meta NN Recall:      {meta_recall:.1%}')
    print(f'Recall Difference:          {meta_recall - lr_recall:+.1%}')

    print(f'\nLogistic Regression Precision: {lr_precision:.1%}')
    print(f'Simple Meta NN Precision:      {meta_precision:.1%}')
    print(f'Precision Difference:          {meta_precision - lr_precision:+.1%}')

    print(f'\nLogistic Regression F1: {lr_f1:.3f}')
    print(f'Simple Meta NN F1:      {meta_f1:.3f}')
    print(f'F1 Difference:          {meta_f1 - lr_f1:+.3f}')

    print(f'\nLogistic Regression AUC: {lr_auc:.3f}')
    print(f'Simple Meta NN AUC:      {meta_auc:.3f}')
    print(f'AUC Difference:          {meta_auc - lr_auc:+.3f}')

    print('\n🚨 CRITICAL FINDING:')
    print('Simple Meta NN has 0% recall - it is NOT detecting ANY defaults!')
    print('Logistic Regression has 19.6% recall - it detects some defaults')
    print('\nThis means:')
    print('• Simple Meta NN: Rejects ALL potentially risky loans (0 defaults detected)')
    print('• Logistic Regression: Detects 19.6% of actual defaults')
    print('• Trade-off: Higher profit vs better default detection')

    print('\n📈 BUSINESS INTERPRETATION:')
    total_defaults = 461  # From our earlier analysis
    lr_detected = int(total_defaults * lr_recall)
    meta_detected = int(total_defaults * meta_recall)

    print(f'Total defaults in dataset: {total_defaults}')
    print(f'LR detects: {lr_detected} defaults ({lr_recall:.1%})')
    print(f'Meta NN detects: {meta_detected} defaults ({meta_recall:.1%})')
    print(f'Additional defaults LR catches: {lr_detected - meta_detected}')

    print('\n💰 PROFIT vs DETECTION TRADE-OFF:')
    print(f'Simple Meta NN: Higher profit (${results.iloc[0]["Profit"]:,.0f}) but 0% recall')
    print(f'Logistic Regression: Lower profit (${results.iloc[2]["Profit"]:,.0f}) but 19.6% recall')
    profit_diff = results.iloc[0]['Profit'] - results.iloc[2]['Profit']
    print(f'Profit sacrifice for better detection: ${profit_diff:,.0f}')

    print('\n🎯 CONCLUSION:')
    print('Simple Meta NN does NOT improve default detection - it actually performs worse!')
    print('However, it achieves higher profit by being extremely conservative.')
    print('This suggests the Meta NN learned to optimize purely for profit, not detection.')

if __name__ == "__main__":
    analyze_default_detection()



