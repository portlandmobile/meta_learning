"""
Feature Engineering for EMBS Dataset
====================================

This module handles feature engineering for the Meta Neural Network,
creating specialized branches for origination and performance features.

Key Features:
- Origination branch: Static loan characteristics and risk categories
- Performance branch: Dynamic performance indicators and delinquency patterns
- Risk-based feature engineering
- Feature scaling and encoding
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Feature engineering for Meta Neural Network branches"""
    
    def __init__(self):
        self.orig_preprocessor = None
        self.perf_preprocessor = None
        self.orig_columns = None
        self.perf_columns = None
        
    def identify_feature_types(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Identify numeric and categorical features for each branch"""
        
        # Origination features (static)
        orig_numeric = [
            'CdtScore', 'UPB', 'InterestRate', 'CLTV', 'LTV', 'DTIRatio',
            'MortInsurancePerc', '#ofunits', 'OrigLoanTerm', '#ofborrows',
            'MetroCode', 'PostalCode', 'PropertyValuation'
        ]
        
        orig_categorical = [
            '1stHomeFlag', 'OccupancySatus', 'Channel', 'PPMflag', 'Amortization',
            'PropertyState', 'PropertyType', 'LoadPurpose', 'SellerName', 'ServiceName',
            'SuperConforming', 'ProgramInd', 'HARPInd', 'InterestOnly', 'MortInsuranceCanellation',
            'CreditRisk', 'DTIRisk', 'CLTVRisk', 'CombinedRisk'
        ]
        
        # Performance features (dynamic) - REMOVED direct delinquency indicators
        perf_numeric = [
            'ActualUPB', 'LoanAge', 'MonthstoMaturity', 'CurrentInterestRate',
            'CurrentUPB', 'MIRecov', 'NonMIRecov', 'Expenses', 'LegalCosts',
            'MaintPresCosts', 'TaxesInsur', 'MiscExpenses', 'ActualLoss',
            'ModCost', 'ELTV', '0balanceUPB', 'CurrentMonthMod', 'InterestBearUPB'
        ]
        
        perf_categorical = [
            'ModFlag', 'balanceCode', 'StepModFlag',
            'DeferredPayment', 'DelinquentDueDisaster', 'BorrowAssist'
        ]
        
        # Filter to only include columns that exist in the data
        orig_numeric = [col for col in orig_numeric if col in data.columns]
        orig_categorical = [col for col in orig_categorical if col in data.columns]
        perf_numeric = [col for col in perf_numeric if col in data.columns]
        perf_categorical = [col for col in perf_categorical if col in data.columns]
        
        return {
            'orig_numeric': orig_numeric,
            'orig_categorical': orig_categorical,
            'perf_numeric': perf_numeric,
            'perf_categorical': perf_categorical
        }
    
    def create_engineered_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create additional engineered features with advanced risk-based calculations"""
        print("🔧 Creating engineered features...")
        
        data_eng = data.copy()
        
        # ========================================================================
        # PHASE 1: ADVANCED RISK-BASED FEATURE ENGINEERING
        # ========================================================================
        
        # ------------------------------------------------------------------------
        # 0. BASELINE FEATURES (needed for other calculations)
        # ------------------------------------------------------------------------
        
        # Loan performance indicators (create these first as they're used later)
        data_eng['UPB_Reduction'] = np.where(
            data_eng['UPB'] > 0,
            (data_eng['UPB'] - data_eng['ActualUPB']) / data_eng['UPB'],
            0
        )
        data_eng['Rate_Change'] = data_eng['CurrentInterestRate'] - data_eng['InterestRate']
        
        # Modification indicators
        data_eng['Has_Modification'] = data_eng['ModFlag'].notna().astype(int)
        data_eng['Has_Assistance'] = data_eng['BorrowAssist'].notna().astype(int)
        
        # ------------------------------------------------------------------------
        # 1. PAYMENT-TO-INCOME RATIOS (Highest Impact)
        # ------------------------------------------------------------------------
        
        # Estimate monthly payment using amortization formula
        # P = L * [r(1+r)^n] / [(1+r)^n - 1]
        # where: L = loan amount, r = monthly rate, n = number of payments
        monthly_rate = (data_eng['InterestRate'] / 100) / 12
        n_payments = data_eng['OrigLoanTerm']
        
        # Calculate monthly payment (handling edge cases)
        data_eng['Monthly_Payment'] = np.where(
            monthly_rate > 0,
            data_eng['UPB'] * (monthly_rate * np.power(1 + monthly_rate, n_payments)) / 
            (np.power(1 + monthly_rate, n_payments) - 1),
            data_eng['UPB'] / n_payments  # If rate is 0, simple division
        )
        
        # Estimate monthly income from DTI ratio
        # DTI = (Monthly_Payment + Other_Debts) / Monthly_Income
        # Assuming Monthly_Payment is roughly 70% of total debt obligations
        data_eng['Estimated_Monthly_Income'] = np.where(
            data_eng['DTIRatio'] > 0,
            (data_eng['Monthly_Payment'] * 100) / (data_eng['DTIRatio'] * 0.7),
            np.nan
        )
        
        # Payment-to-Income Ratio (more precise than DTI alone)
        data_eng['Payment_to_Income_Ratio'] = np.where(
            data_eng['Estimated_Monthly_Income'] > 0,
            (data_eng['Monthly_Payment'] / data_eng['Estimated_Monthly_Income']) * 100,
            data_eng['DTIRatio'] * 0.7  # Fallback estimate
        )
        
        # Total Housing Expense Ratio (including taxes & insurance)
        data_eng['Housing_Expense_Ratio'] = np.where(
            data_eng['Estimated_Monthly_Income'] > 0,
            ((data_eng['Monthly_Payment'] + data_eng['TaxesInsur']) / 
             data_eng['Estimated_Monthly_Income']) * 100,
            data_eng['DTIRatio']
        )
        
        # Residual income after housing (measure of financial cushion)
        data_eng['Residual_Income_Ratio'] = (
            100 - data_eng['Payment_to_Income_Ratio']
        )
        
        # ------------------------------------------------------------------------
        # 2. FINANCIAL STRESS INDICATORS
        # ------------------------------------------------------------------------
        
        # Equity cushion in dollars (buffer before underwater)
        data_eng['Equity_Cushion_Dollars'] = (
            (100 - data_eng['CLTV']) / 100 * data_eng['PropertyValuation']
        )
        
        # Equity cushion as percentage of UPB
        data_eng['Equity_Cushion_Pct'] = np.where(
            data_eng['UPB'] > 0,
            (data_eng['Equity_Cushion_Dollars'] / data_eng['UPB']) * 100,
            0
        )
        
        # Payment shock percentage (if rate has changed)
        data_eng['Payment_Shock_Pct'] = np.where(
            data_eng['InterestRate'] > 0,
            ((data_eng['CurrentInterestRate'] - data_eng['InterestRate']) / 
             data_eng['InterestRate']) * 100,
            0
        )
        
        # Debt burden score (cumulative stress factors)
        data_eng['Debt_Burden_Score'] = (
            (data_eng['DTIRatio'] > 43).astype(int) +
            (data_eng['CLTV'] > 80).astype(int) +
            (data_eng['CdtScore'] < 680).astype(int) +
            (data_eng['InterestRate'] > 6.0).astype(int)
        )
        
        # Financial flexibility (months of cushion available)
        data_eng['Financial_Flexibility_Months'] = np.where(
            data_eng['Monthly_Payment'] > 0,
            data_eng['Equity_Cushion_Dollars'] / (data_eng['Monthly_Payment'] * 6),
            0
        )
        
        # Underwater flag (CLTV > 100%)
        data_eng['Is_Underwater'] = (data_eng['CLTV'] > 100).astype(int)
        
        # Near underwater flag (CLTV 95-100%)
        data_eng['Near_Underwater'] = (
            (data_eng['CLTV'] > 95) & (data_eng['CLTV'] <= 100)
        ).astype(int)
        
        # ------------------------------------------------------------------------
        # 3. WEIGHTED RISK SCORES (Business-Impact Weighted)
        # ------------------------------------------------------------------------
        
        # Weighted risk score (higher weight = higher business impact)
        data_eng['Weighted_Risk_Score'] = (
            (data_eng['CdtScore'] < 680).astype(int) * 3.0 +      # Credit risk (highest)
            (data_eng['CdtScore'] < 620).astype(int) * 2.0 +      # Subprime (extra penalty)
            (data_eng['DTIRatio'] > 43).astype(int) * 2.0 +       # High DTI
            (data_eng['DTIRatio'] > 50).astype(int) * 1.0 +       # Very high DTI (extra)
            (data_eng['CLTV'] > 90).astype(int) * 2.5 +           # High CLTV
            (data_eng['CLTV'] > 100).astype(int) * 2.0 +          # Underwater (extra)
            (data_eng['ModFlag'].notna()).astype(int) * 1.5 +     # Has modification
            (data_eng['InterestRate'] > 6.0).astype(int) * 1.0 +  # High rate
            (data_eng['InterestRate'] > 7.0).astype(int) * 0.5    # Very high rate (extra)
        )
        
        # Compounding risk factors (multiplicative effects)
        data_eng['Compounding_Risk_Score'] = (
            # Poor credit + High debt = Major risk
            ((data_eng['CdtScore'] < 680) & (data_eng['DTIRatio'] > 43)).astype(int) * 5.0 +
            # No equity + High debt = Trouble
            ((data_eng['CLTV'] > 90) & (data_eng['DTIRatio'] > 43)).astype(int) * 4.0 +
            # Poor credit + No equity = High default risk
            ((data_eng['CdtScore'] < 680) & (data_eng['CLTV'] > 90)).astype(int) * 4.0 +
            # Triple threat: Poor credit + High debt + No equity
            ((data_eng['CdtScore'] < 680) & (data_eng['DTIRatio'] > 43) & 
             (data_eng['CLTV'] > 90)).astype(int) * 8.0
        )
        
        # Risk trajectory (improving or deteriorating?)
        data_eng['Risk_Trajectory_Score'] = (
            (data_eng['UPB_Reduction'] > 0).astype(int) * -1.0 +      # Paying down = good
            (data_eng['UPB_Reduction'] < -0.05).astype(int) * 2.0 +   # Increasing UPB = bad
            (data_eng['Rate_Change'] > 0).astype(int) * 1.0 +         # Rate increased = bad
            (data_eng['Rate_Change'] > 1.0).astype(int) * 1.0 +       # Big rate increase = worse
            (data_eng['ModFlag'].notna()).astype(int) * 2.0 +         # Modified = distress
            (data_eng['BorrowAssist'].notna()).astype(int) * 1.0      # Assistance = distress
        )
        
        # Combined ultimate risk score (weighted sum of all factors)
        data_eng['Ultimate_Risk_Score'] = (
            data_eng['Weighted_Risk_Score'] * 0.4 +
            data_eng['Compounding_Risk_Score'] * 0.3 +
            data_eng['Debt_Burden_Score'] * 0.2 +
            (data_eng['Risk_Trajectory_Score'].clip(0, 10)) * 0.1
        )
        
        # ------------------------------------------------------------------------
        # 4. BEHAVIORAL PERFORMANCE PATTERNS
        # ------------------------------------------------------------------------
        
        # Payment velocity (rate of UPB reduction per month)
        data_eng['Payment_Velocity'] = np.where(
            data_eng['LoanAge'] > 0,
            (data_eng['UPB'] - data_eng['ActualUPB']) / data_eng['LoanAge'],
            0
        )
        
        # Interest burden ratio
        current_monthly_interest = (data_eng['CurrentInterestRate'] / 100 / 12) * data_eng['CurrentUPB']
        data_eng['Interest_Burden_Ratio'] = np.where(
            data_eng['Monthly_Payment'] > 0,
            current_monthly_interest / data_eng['Monthly_Payment'] * 100,
            0
        )
        
        # Distress signal (composite of modification/assistance indicators)
        data_eng['Distress_Signal_Score'] = (
            (data_eng['ModFlag'].notna()).astype(int) +
            (data_eng['BorrowAssist'].notna()).astype(int) +
            (data_eng['ModCost'] > 0).astype(int) +
            (data_eng['DeferredPayment'].notna()).astype(int)
        )
        
        # Expected loss severity (if default occurs, estimated loss)
        data_eng['Expected_Loss_Severity'] = np.where(
            data_eng['ELTV'] > 80,
            np.maximum(data_eng['ELTV'] - 80, 0) / 100 * data_eng['ActualUPB'],
            0
        )
        
        # Loss severity ratio (as % of original UPB)
        data_eng['Loss_Severity_Ratio'] = np.where(
            data_eng['UPB'] > 0,
            data_eng['Expected_Loss_Severity'] / data_eng['UPB'] * 100,
            0
        )
        
        # ------------------------------------------------------------------------
        # 5. ADDITIONAL COMPLEMENTARY FEATURES
        # ------------------------------------------------------------------------
        
        # Financial ratios and interactions
        data_eng['DTI_CLTV_Interaction'] = data_eng['DTIRatio'] * data_eng['CLTV']
        data_eng['Credit_to_Equity_Ratio'] = data_eng['CdtScore'] / (100 - data_eng['LTV'] + 1e-6)
        data_eng['Payment_Burden'] = data_eng['InterestRate'] * data_eng['UPB'] / 1000
        data_eng['Equity_Position'] = data_eng['ELTV'] - data_eng['LTV']
        
        # Risk score combinations (simple count for comparison)
        data_eng['High_Risk_Score'] = (
            (data_eng['CreditRisk'] == 'Subprime').astype(int) +
            (data_eng['DTIRisk'] == 'Very High').astype(int) +
            (data_eng['CLTVRisk'] == 'Very High').astype(int)
        )
        
        print("✅ Created advanced risk-based engineered features")
        print(f"   • Payment-to-Income ratios and affordability metrics")
        print(f"   • Financial stress indicators and equity cushions")
        print(f"   • Weighted and compounding risk scores")
        print(f"   • Behavioral patterns and distress signals")
        
        return data_eng
    
    def create_preprocessors(self, data: pd.DataFrame) -> Tuple[ColumnTransformer, ColumnTransformer]:
        """Create preprocessing pipelines for origination and performance branches"""
        print("🔧 Creating preprocessing pipelines...")
        
        feature_types = self.identify_feature_types(data)
        
        # Convert categorical columns to strings to avoid mixed type issues
        data_clean = data.copy()
        for col in feature_types['orig_categorical'] + feature_types['perf_categorical']:
            if col in data_clean.columns:
                data_clean[col] = data_clean[col].astype(str)
        
        # Origination preprocessor
        orig_numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        orig_categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.orig_preprocessor = ColumnTransformer([
            ('num', orig_numeric_pipeline, feature_types['orig_numeric']),
            ('cat', orig_categorical_pipeline, feature_types['orig_categorical'])
        ])
        
        # Performance preprocessor
        perf_numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        perf_categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.perf_preprocessor = ColumnTransformer([
            ('num', perf_numeric_pipeline, feature_types['perf_numeric']),
            ('cat', perf_categorical_pipeline, feature_types['perf_categorical'])
        ])
        
        # Store column names for later use
        self.orig_columns = feature_types['orig_numeric'] + feature_types['orig_categorical']
        self.perf_columns = feature_types['perf_numeric'] + feature_types['perf_categorical']
        
        print("✅ Created preprocessing pipelines")
        return self.orig_preprocessor, self.perf_preprocessor
    
    def fit_transform_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fit preprocessors and transform training and test data"""
        print("🔄 Fitting and transforming features...")
        
        # Create engineered features
        X_train_eng = self.create_engineered_features(X_train)
        X_test_eng = self.create_engineered_features(X_test)
        
        # Create preprocessors
        self.create_preprocessors(X_train_eng)
        
        # Convert categorical columns to strings for both train and test
        feature_types = self.identify_feature_types(X_train_eng)
        for col in feature_types['orig_categorical'] + feature_types['perf_categorical']:
            if col in X_train_eng.columns:
                X_train_eng[col] = X_train_eng[col].astype(str)
            if col in X_test_eng.columns:
                X_test_eng[col] = X_test_eng[col].astype(str)
        
        # Transform origination features
        X_train_orig = self.orig_preprocessor.fit_transform(
            X_train_eng[self.orig_columns]
        )
        X_test_orig = self.orig_preprocessor.transform(
            X_test_eng[self.orig_columns]
        )
        
        # Transform performance features
        X_train_perf = self.perf_preprocessor.fit_transform(
            X_train_eng[self.perf_columns]
        )
        X_test_perf = self.perf_preprocessor.transform(
            X_test_eng[self.perf_columns]
        )
        
        print(f"✅ Transformed features:")
        print(f"   Origination: {X_train_orig.shape[1]} features")
        print(f"   Performance: {X_train_perf.shape[1]} features")
        
        return X_train_orig, X_test_orig, X_train_perf, X_test_perf
    
    def get_feature_names(self) -> Tuple[List[str], List[str]]:
        """Get feature names for origination and performance branches"""
        if self.orig_preprocessor is None or self.perf_preprocessor is None:
            raise ValueError("Preprocessors not fitted yet. Call fit_transform_features first.")
        
        # Get feature names from preprocessors
        orig_names = self.orig_preprocessor.get_feature_names_out()
        perf_names = self.perf_preprocessor.get_feature_names_out()
        
        return orig_names.tolist(), perf_names.tolist()

def create_business_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create business-specific features for profit optimization"""
    print("💰 Creating business features...")
    
    data_biz = data.copy()
    
    # Loan size categories
    data_biz['LoanSize_Category'] = pd.cut(
        data_biz['UPB'], 
        bins=[0, 150000, 300000, 500000, float('inf')], 
        labels=['Small', 'Medium', 'Large', 'Jumbo']
    )
    
    # Interest rate categories
    data_biz['Rate_Category'] = pd.cut(
        data_biz['InterestRate'], 
        bins=[0, 4.5, 5.5, 6.5, float('inf')], 
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    # Risk-based pricing indicators
    data_biz['High_Risk_Pricing'] = (
        (data_biz['InterestRate'] > 6.0) & 
        (data_biz['CombinedRisk'].isin(['High', 'Very High']))
    ).astype(int)
    
    # Profitability indicators
    data_biz['Expected_Revenue'] = data_biz['UPB'] * 0.13  # 13% revenue assumption
    data_biz['Expected_Loss'] = data_biz['UPB'] * 0.16 * data_biz['Default']  # 16% loss if default
    
    print("✅ Created business features")
    return data_biz

def main():
    """Test the feature engineer"""
    from data_loader import DataConfig, EMBSDataLoader
    
    # Load data
    config = DataConfig()
    loader = EMBSDataLoader(config)
    orig_data, perf_data, merged_data = loader.load_and_preprocess()
    
    # Test feature engineering
    engineer = FeatureEngineer()
    
    # Split data for testing
    from sklearn.model_selection import train_test_split
    X = merged_data.drop('Default', axis=1)
    y = merged_data['Default']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Transform features
    X_train_orig, X_test_orig, X_train_perf, X_test_perf = engineer.fit_transform_features(X_train, X_test)
    
    print(f"\n📊 Feature Engineering Results:")
    print(f"Training origination features: {X_train_orig.shape}")
    print(f"Training performance features: {X_train_perf.shape}")
    print(f"Test origination features: {X_test_orig.shape}")
    print(f"Test performance features: {X_test_perf.shape}")

if __name__ == "__main__":
    main()
