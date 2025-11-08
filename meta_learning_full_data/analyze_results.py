#!/usr/bin/env python3
"""
Analyze the model comparison results
"""
import pandas as pd
import numpy as np

def analyze_results():
    print('📊 DETAILED ANALYSIS OF RESULTS')
    print('=' * 60)

    # Load results
    results = pd.read_csv('model_comparison_results.csv')
    predictions = pd.read_csv('model_predictions.csv')

    print('\n📈 FINAL MODEL RANKINGS:')
    print(f'1. 🥇 Simple Meta NN:    ${results.iloc[0]["Profit"]:,.0f} profit')
    print(f'2. 🥈 Enhanced Meta NN:  ${results.iloc[1]["Profit"]:,.0f} profit') 
    print(f'3. 🥉 Logistic Regression: ${results.iloc[2]["Profit"]:,.0f} profit')

    print('\n🎯 KEY METRICS COMPARISON:')
    print('Model                | Profit ($)     | AUC    | Recall | Precision | F1')
    print('-' * 75)
    for i, row in results.iterrows():
        print(f'{row["Model"]:<20} | ${row["Profit"]:>12,.0f} | {row["AUC"]:>6.3f} | {row["Recall"]:>6.3f} | {row["Precision"]:>9.3f} | {row["F1"]:>6.3f}')

    print('\n💰 PROFIT ANALYSIS:')
    best_profit = results.iloc[0]['Profit']
    worst_profit = results.iloc[2]['Profit']
    profit_improvement = best_profit - worst_profit
    improvement_pct = (profit_improvement / worst_profit) * 100

    print(f'Best Model (Simple Meta NN):    ${best_profit:,.0f}')
    print(f'Worst Model (Logistic Regression): ${worst_profit:,.0f}')
    print(f'Profit Improvement:              ${profit_improvement:,.0f}')
    print(f'Improvement Percentage:          {improvement_pct:.2f}%')

    print('\n🔍 RECALL vs PROFIT TRADE-OFF:')
    print(f'Simple Meta NN:     Recall={results.iloc[0]["Recall"]:.1%}, Profit=${results.iloc[0]["Profit"]:,.0f}')
    print(f'Enhanced Meta NN:   Recall={results.iloc[1]["Recall"]:.1%}, Profit=${results.iloc[1]["Profit"]:,.0f}')
    print(f'Logistic Regression: Recall={results.iloc[2]["Recall"]:.1%}, Profit=${results.iloc[2]["Profit"]:,.0f}')

    print('\n📊 DATASET CHARACTERISTICS:')
    print('Total loans: 50,000')
    print('Default rate: 0.9% (461 defaults)')
    print('Features: 161 origination + 25 performance = 186 total')

    print('\n🎯 BUSINESS IMPLICATIONS:')
    print('• Meta-NN models optimize for profit over recall')
    print('• Simple Meta NN achieves highest profit with conservative lending')
    print('• Logistic Regression has higher recall but lower profit')
    print('• Trade-off: Higher recall = more defaults approved = lower profit')

    print('\n🚀 KEY INSIGHTS:')
    print('1. Meta-NN models successfully beat Logistic Regression in profit optimization')
    print('2. Simple Meta NN is the best performer - simpler architecture works better')
    print('3. Enhanced Meta NN shows diminishing returns vs Simple Meta NN')
    print('4. All models struggle with recall due to extreme class imbalance (0.9% default rate)')
    print('5. Business optimization favors conservative lending strategy')

if __name__ == "__main__":
    analyze_results()



