# Meta Learning Full Data Analysis

## Overview
This project implements a comprehensive comparison between Logistic Regression and Meta Neural Networks using the full EMBS dataset with both origination and performance data.

## Dataset
- **Origination Data**: 50,000 loans with 32 static features
- **Performance Data**: 969,328 monthly records with 32 dynamic features
- **Common Key**: LoanSeqNumber (PYYQnXXXXXXX format)
- **Default Definition**: DefaultStatus >= 2 (60+ days delinquent)

## Key Features

### Origination Features (Static)
- **Financial Risk**: DTI, CLTV, LTV, Credit Score, Interest Rate, UPB
- **Property**: State, Type, Metro Code, Postal Code
- **Borrower**: Number of borrowers, First-time buyer, Occupancy status
- **Channel**: Origination channel, Seller, Servicer

### Performance Features (Dynamic)
- **Delinquency**: Default Status, Loan Age, Delinquent Interest
- **Financial**: Actual UPB, Current Interest Rate, ELTV
- **Modifications**: Modification flags and costs
- **Assistance**: Borrower assistance programs

## Models Compared
1. **Logistic Regression**: Baseline linear model
2. **Simple Meta-NN**: Basic two-branch architecture
3. **Enhanced Meta-NN**: Advanced architecture with attention, batch norm, etc.

## Business Objectives
- **Profit Maximization**: Optimize for business revenue vs losses
- **Recall Optimization**: Catch more actual defaults
- **Risk Stratification**: Create risk categories based on DTI/CLTV/Credit Score

## Usage
```python
python main_comparison.py
```

## Files
- `data_loader.py`: Data loading and preprocessing
- `feature_engineering.py`: Risk categories and feature creation
- `models.py`: Model implementations
- `business_evaluation.py`: Profit optimization functions
- `main_comparison.py`: Main comparison script

