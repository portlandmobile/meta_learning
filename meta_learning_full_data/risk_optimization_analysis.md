# Risk Optimization Analysis: Logistic Regression vs Meta-NN Models

## Executive Summary

We successfully completed a risk-optimized comparison focusing on **default detection (recall)** rather than pure profitability. The models were trained using **Focal Loss** and **class-weighted Logistic Regression** to prioritize identifying high-risk loans.

## Key Findings

### 🎯 Default Detection Performance

| Model | Recall | Precision | F1 Score | AUC | Defaults Detected |
|-------|--------|-----------|----------|-----|-------------------|
| **Logistic Regression** | 98.9% | 1.03% | 2.03% | 0.802 | 91/92 defaults |
| **Simple Meta NN** | 100.0% | 0.94% | 1.87% | 0.798 | 92/92 defaults |
| **Enhanced Meta NN** | 100.0% | 0.92% | 1.82% | 0.509 | 92/92 defaults |

### 💰 Business Impact Analysis

| Model | Approved Loans | Revenue | Losses | Net Profit |
|-------|---------------|---------|--------|------------|
| **Logistic Regression** | 1,128 (11.3%) | $18.33M | $23K | **$18.31M** |
| **Simple Meta NN** | 262 (2.6%) | $4.26M | $0 | **$4.26M** |
| **Enhanced Meta NN** | 0 (0.0%) | $0 | $0 | **$0** |

## Detailed Analysis

### 1. Default Detection Excellence
- **Perfect Detection**: Both Meta-NN models achieved 100% recall (detected all 92 defaults)
- **Near-Perfect**: Logistic Regression detected 98.9% (91/92 defaults)
- **Zero False Negatives**: Meta-NN models had no missed defaults

### 2. Conservative vs Aggressive Strategies

#### Logistic Regression: Balanced Approach
- **Moderate Conservatism**: Approved 11.3% of loans
- **High Revenue**: Generated $18.33M in revenue
- **Acceptable Risk**: Only 1 missed default out of 92
- **Best AUC**: 0.802 (strongest overall discrimination)

#### Simple Meta NN: Ultra-Conservative
- **Extreme Conservatism**: Approved only 2.6% of loans
- **Lower Revenue**: $4.26M but zero losses
- **Perfect Safety**: 100% default detection with no losses
- **Good AUC**: 0.798 (competitive discrimination)

#### Enhanced Meta NN: Overly Conservative
- **Reject-All Strategy**: Approved 0% of loans
- **Zero Revenue**: No approved loans, no revenue
- **Perfect Safety**: 100% default detection but unusable for business
- **Poor AUC**: 0.509 (near-random performance)

### 3. Risk-Return Trade-offs

| Strategy | Risk Level | Revenue | Profit | Business Viability |
|----------|------------|---------|--------|-------------------|
| **Logistic Regression** | Moderate | High | **Highest** | ✅ **Recommended** |
| **Simple Meta NN** | Ultra-Low | Medium | Medium | ⚠️ Overly Conservative |
| **Enhanced Meta NN** | Zero | None | None | ❌ Not Viable |

## Key Insights

### 1. **Logistic Regression Wins for Balanced Risk Management**
- Achieved near-perfect default detection (98.9%) with reasonable revenue
- Only missed 1 default while generating significant profit
- Best overall discrimination (AUC = 0.802)

### 2. **Meta-NN Models Are Overly Conservative**
- While achieving perfect default detection, they're too conservative for business use
- Simple Meta-NN approved only 2.6% of loans
- Enhanced Meta-NN rejected all loans (0% approval rate)

### 3. **Focal Loss Impact**
- Successfully prioritized default detection over general accuracy
- Models focused on identifying the rare default cases (0.9% of dataset)
- Led to very conservative lending policies

### 4. **Business Recommendation**
For a **risk-focused lending strategy**, **Logistic Regression** provides the optimal balance:
- **High default detection** (98.9% recall)
- **Reasonable approval rates** (11.3%)
- **Strong profitability** ($18.31M profit)
- **Manageable risk** (only 1 missed default)

## Technical Observations

### 1. **Data Characteristics**
- **Extreme Class Imbalance**: Only 0.9% default rate
- **Challenging Detection**: Very rare default events
- **Historical Performance Data**: Used 6-month-old performance data to avoid data leakage

### 2. **Model Behavior**
- **Meta-NN Overfitting**: Enhanced Meta-NN learned to reject everything
- **Simple Meta-NN Learning**: Achieved perfect recall but extreme conservatism
- **Logistic Regression Stability**: Maintained good balance across metrics

### 3. **Feature Engineering Impact**
- **Two-Branch Architecture**: Origination + Performance features
- **Risk Categories**: DTI, CLTV, Credit Score ranges
- **Time-Based Features**: Historical vs current performance data

## Conclusion

While **Meta-NN models achieved perfect default detection**, they proved **too conservative for practical business use**. The **Logistic Regression model** emerged as the clear winner by providing:

1. **Excellent default detection** (98.9% recall)
2. **Reasonable business volume** (11.3% approval rate)
3. **Strong profitability** ($18.31M profit)
4. **Acceptable risk tolerance** (1 missed default)

For a **risk-optimized lending strategy**, **Logistic Regression with class weighting** provides the optimal balance between default detection and business viability.



