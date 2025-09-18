#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[3]:


from sklearn.model_selection import train_test_split


# In[6]:


from sklearn.preprocessing import StandardScaler
#import matplotlib.pyplot as plt


# In[7]:


def generate_credit_risk_dataset(n_samples=5000, random_state=42):
    """
    Generate a realistic credit risk dataset with two distinct feature groups:
    - Financial features (income, debt, credit history)
    - Behavioral features (spending patterns, payment history)
    """
    np.random.seed(random_state)
    
    # Financial Features (10 features) - Higher impact on default risk
    annual_income = np.random.lognormal(10.5, 0.5, n_samples)  # $30k-$200k range
    debt_to_income = np.random.beta(2, 5, n_samples) * 0.8  # 0-80% DTI
    credit_score = np.random.normal(650, 100, n_samples)
    credit_score = np.clip(credit_score, 300, 850)
    years_credit_history = np.random.exponential(8, n_samples)
    years_credit_history = np.clip(years_credit_history, 0, 30)
    num_credit_accounts = np.random.poisson(5, n_samples)
    mortgage_amount = np.random.exponential(200000, n_samples) * (np.random.random(n_samples) < 0.6)
    savings_ratio = np.random.beta(1, 3, n_samples) * 0.3  # 0-30% savings rate
    employment_years = np.random.exponential(5, n_samples)
    employment_years = np.clip(employment_years, 0, 40)
    loan_amount = np.random.lognormal(9, 1, n_samples)  # Loan amount requested
    collateral_value = loan_amount * (0.8 + 0.4 * np.random.random(n_samples))
    
    # Behavioral Features (10 features) - Moderate impact on default risk
    monthly_transactions = np.random.poisson(50, n_samples)
    avg_transaction_amount = np.random.lognormal(3, 0.8, n_samples)
    late_payments_12m = np.random.poisson(2, n_samples)
    credit_utilization = np.random.beta(2, 3, n_samples)
    num_bank_accounts = np.random.poisson(3, n_samples) + 1
    online_banking_usage = np.random.beta(3, 2, n_samples)
    payment_method_diversity = np.random.poisson(4, n_samples) + 1
    financial_app_usage = np.random.beta(2, 3, n_samples)
    investment_accounts = np.random.poisson(1, n_samples)
    insurance_policies = np.random.poisson(2, n_samples) + 1
    
    # Combine all features
    financial_features = np.column_stack([
        annual_income, debt_to_income, credit_score, years_credit_history,
        num_credit_accounts, mortgage_amount, savings_ratio, employment_years,
        loan_amount, collateral_value
    ])
    
    behavioral_features = np.column_stack([
        monthly_transactions, avg_transaction_amount, late_payments_12m,
        credit_utilization, num_bank_accounts, online_banking_usage,
        payment_method_diversity, financial_app_usage, investment_accounts,
        insurance_policies
    ])
    
    # Create target variable with different weights for each feature group
    # Financial features have 3x more impact than behavioral features
    financial_score = np.sum((financial_features - np.mean(financial_features, axis=0)) / np.std(financial_features, axis=0), axis=1)
    behavioral_score = np.sum((behavioral_features - np.mean(behavioral_features, axis=0)) / np.std(behavioral_features, axis=0), axis=1)
    
    # Default risk: higher financial risk + moderate behavioral risk
    risk_score = 3.0 * financial_score + 1.0 * behavioral_score
    default_probability = 1 / (1 + np.exp(-risk_score / 5))
    y = (np.random.random(n_samples) < default_probability).astype(int)
    
    # Combine features
    X = np.column_stack([financial_features, behavioral_features])
    
    # Create feature names
    financial_names = [
        'annual_income', 'debt_to_income_ratio', 'credit_score', 'years_credit_history',
        'num_credit_accounts', 'mortgage_amount', 'savings_ratio', 'employment_years',
        'loan_amount_requested', 'collateral_value'
    ]
    
    behavioral_names = [
        'monthly_transactions', 'avg_transaction_amount', 'late_payments_12m',
        'credit_utilization', 'num_bank_accounts', 'online_banking_usage',
        'payment_method_diversity', 'financial_app_usage', 'investment_accounts',
        'insurance_policies'
    ]
    
    feature_names = financial_names + behavioral_names
    
    # Create DataFrame for easy inspection
    df = pd.DataFrame(X, columns=feature_names)
    df['default_risk'] = y
    
    return X, y, feature_names, df

def prepare_data_for_models(X, y, test_size=0.2, random_state=42):
    """
    Prepare data for both monolithic and meta-learning models
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Split features for meta-learning model
    # First 10: Financial features, Last 10: Behavioral features
    X_train_financial = X_train_scaled[:, :10]
    X_train_behavioral = X_train_scaled[:, 10:]
    X_test_financial = X_test_scaled[:, :10]
    X_test_behavioral = X_test_scaled[:, 10:]
    
    return {
        'monolithic': {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test
        },
        'meta_learning': {
            'X_train': [X_train_financial, X_train_behavioral],
            'X_test': [X_test_financial, X_test_behavioral],
            'y_train': y_train,
            'y_test': y_test
        }
    }

# Generate the dataset
if __name__ == "__main__":
    print("Generating credit risk dataset...")
    X, y, feature_names, df = generate_credit_risk_dataset(n_samples=5000)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Default rate: {np.mean(y):.1%}")
    print(f"\nFeature groups:")
    print(f"Financial features (0-9): {feature_names[:10]}")
    print(f"Behavioral features (10-19): {feature_names[10:]}")
    
    # Display basic statistics
    print(f"\nDataset summary:")
    print(df.describe())
    
    # Prepare data for both models
    data = prepare_data_for_models(X, y)
    
    print(f"\nTraining set size: {data['monolithic']['X_train'].shape[0]}")
    print(f"Test set size: {data['monolithic']['X_test'].shape[0]}")
    print(f"Features are standardized and ready for training!")
    
    # Save to CSV for easy loading
    df.to_csv('credit_risk_dataset.csv', index=False)
    print(f"\nDataset saved to 'credit_risk_dataset.csv'")


# In[ ]:




