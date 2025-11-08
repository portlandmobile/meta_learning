"""
Data Loading and Preprocessing for EMBS Dataset
==============================================

This module handles loading and preprocessing of the origination and performance datasets
for loan default prediction using Meta Neural Networks.

Key Features:
- Loads origination and performance data
- Handles missing values (999/9999 as missing)
- Creates risk categories based on DTI/CLTV/Credit Score
- Merges datasets on LoanSeqNumber
- Uses latest performance record per loan
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing"""
    # File paths
    orig_data_path: str = "/Users/peekay/Downloads/sample_2023/sample_orig_2023.csv"
    perf_data_path: str = "/Users/peekay/Downloads/sample_2023/sample_svcg_2023.csv"
    schema_path: str = "/Users/peekay/Downloads/sample_2023/EMBS data columns.xlsx"
    
    # Data parameters
    test_size: float = 0.2
    random_state: int = 42
    
    # Default definition
    default_threshold: int = 2  # DefaultStatus >= 2 (60+ days delinquent)
    
    # Missing value indicators
    missing_indicators: List[int] = None
    
    def __post_init__(self):
        if self.missing_indicators is None:
            self.missing_indicators = [999, 9999, 99, 9]

class EMBSDataLoader:
    """Main class for loading and preprocessing EMBS data"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.orig_schema = None
        self.perf_schema = None
        self.orig_data = None
        self.perf_data = None
        self.merged_data = None
        
    def load_schemas(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load column schemas from Excel file"""
        print("📋 Loading data schemas...")
        
        # Load schemas
        self.orig_schema = pd.read_excel(
            self.config.schema_path, 
            sheet_name='Origination full schema'
        )
        self.perf_schema = pd.read_excel(
            self.config.schema_path, 
            sheet_name='Monthly Perf full schema'
        )
        
        print(f"✅ Loaded schemas: {len(self.orig_schema)} origination columns, {len(self.perf_schema)} performance columns")
        return self.orig_schema, self.perf_schema
    
    def load_origination_data(self) -> pd.DataFrame:
        """Load and preprocess origination data"""
        print("📊 Loading origination data...")
        
        # Load raw data (pipe-delimited with headers)
        orig_raw = pd.read_csv(
            self.config.orig_data_path, 
            sep='|',
            low_memory=False
        )
        
        print(f"✅ Loaded {len(orig_raw):,} origination records")
        return orig_raw
    
    def load_performance_data(self) -> pd.DataFrame:
        """Load and preprocess performance data"""
        print("📊 Loading performance data...")
        
        # Load raw data (pipe-delimited with headers)
        perf_raw = pd.read_csv(
            self.config.perf_data_path, 
            sep='|',
            low_memory=False
        )
        
        print(f"✅ Loaded {len(perf_raw):,} performance records")
        return perf_raw
    
    def get_historical_performance(self, perf_data: pd.DataFrame, months_back: int = 6) -> pd.DataFrame:
        """Get performance record from N months ago for each loan (OPTIMIZED)"""
        print(f"🔄 Getting performance records from {months_back} months ago...")
        
        # Convert MonthlyReportPeriod to datetime for sorting
        perf_data['ReportDate'] = pd.to_datetime(
            perf_data['MonthlyReportPeriod'].astype(str), 
            format='%Y%m', 
            errors='coerce'
        )
        
        # Sort by loan and date
        perf_data_sorted = perf_data.sort_values(['LoanSeqNumber', 'ReportDate'])
        
        # OPTIMIZED: Use vectorized groupby operations instead of Python for loop
        # Get count of records per loan
        record_counts = perf_data_sorted.groupby('LoanSeqNumber').size()
        
        # Split into two groups: enough history vs not enough
        enough_history = record_counts[record_counts > months_back].index
        not_enough_history = record_counts[record_counts <= months_back].index
        
        # For loans with enough history, get the record at -(months_back + 1)
        hist_enough = perf_data_sorted[perf_data_sorted['LoanSeqNumber'].isin(enough_history)]\
            .groupby('LoanSeqNumber', group_keys=False)\
            .nth(-(months_back + 1))\
            .reset_index(drop=True)
        
        # For loans without enough history, get the first record
        hist_not_enough = perf_data_sorted[perf_data_sorted['LoanSeqNumber'].isin(not_enough_history)]\
            .groupby('LoanSeqNumber', group_keys=False)\
            .nth(0)\
            .reset_index(drop=True)
        
        # Combine both groups
        historical_df = pd.concat([hist_enough, hist_not_enough], ignore_index=True)
        
        # Drop the temporary date column
        historical_df = historical_df.drop('ReportDate', axis=1)
        
        print(f"✅ Got historical performance for {len(historical_df):,} loans (FAST!)")
        return historical_df
    
    def get_latest_performance(self, perf_data: pd.DataFrame) -> pd.DataFrame:
        """Get latest performance record for each loan (for target variable)"""
        print("🔄 Getting latest performance records for target variable...")
        
        # Convert MonthlyReportPeriod to datetime for sorting
        perf_data['ReportDate'] = pd.to_datetime(
            perf_data['MonthlyReportPeriod'].astype(str), 
            format='%Y%m', 
            errors='coerce'
        )
        
        # Sort by loan and date, then get latest record per loan
        latest_perf = perf_data.sort_values(['LoanSeqNumber', 'ReportDate'])\
                              .groupby('LoanSeqNumber')\
                              .tail(1)\
                              .reset_index(drop=True)
        
        # Drop the temporary date column
        latest_perf = latest_perf.drop('ReportDate', axis=1)
        
        print(f"✅ Got latest performance for {len(latest_perf):,} loans")
        return latest_perf
    
    def handle_missing_values(self, data: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Handle missing values based on validation rules"""
        print(f"🔧 Handling missing values for {data_type} data...")
        
        data_clean = data.copy()
        
        # Get schema for this data type
        schema = self.orig_schema if data_type == 'origination' else self.perf_schema
        
        for _, row in schema.iterrows():
            col_name = row['Column Name']
            valid_values = row['VALID VALUES/CALCULATIONS']
            
            if col_name not in data_clean.columns:
                continue
                
            # Handle different missing value patterns
            if '999' in str(valid_values) or 'Not Available' in str(valid_values):
                # Replace 999/9999 with NaN
                data_clean[col_name] = data_clean[col_name].replace(
                    self.config.missing_indicators, np.nan
                )
            
            # Handle specific cases
            if col_name in ['CdtScore', 'CLTV', 'LTV', 'DTIRatio']:
                # These have 999/9999 as "Not Available"
                data_clean[col_name] = data_clean[col_name].replace(
                    [999, 9999], np.nan
                )
            elif col_name in ['MortInsurancePerc', '#ofunits', '#ofborrows']:
                # These have 99 as "Not Available"
                data_clean[col_name] = data_clean[col_name].replace(99, np.nan)
            elif col_name in ['1stHomeFlag', 'OccupancySatus', 'Channel', 'PropertyType']:
                # These have 9 as "Not Available"
                data_clean[col_name] = data_clean[col_name].replace('9', np.nan)
        
        print(f"✅ Handled missing values for {data_type} data")
        return data_clean
    
    def create_risk_categories(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create risk categories based on DTI/CLTV/Credit Score"""
        print("🎯 Creating risk categories...")
        
        data_with_risk = data.copy()
        
        # Credit Score Risk Categories
        data_with_risk['CreditRisk'] = pd.cut(
            data_with_risk['CdtScore'], 
            bins=[0, 620, 680, 740, 850], 
            labels=['Subprime', 'Near Prime', 'Prime', 'Super Prime'],
            include_lowest=True
        )
        
        # DTI Risk Categories
        data_with_risk['DTIRisk'] = pd.cut(
            data_with_risk['DTIRatio'], 
            bins=[0, 28, 36, 43, 65], 
            labels=['Low', 'Moderate', 'High', 'Very High'],
            include_lowest=True
        )
        
        # CLTV Risk Categories
        data_with_risk['CLTVRisk'] = pd.cut(
            data_with_risk['CLTV'], 
            bins=[0, 60, 80, 90, 100], 
            labels=['Low', 'Moderate', 'High', 'Very High'],
            include_lowest=True
        )
        
        # Combined Risk Score (0-3 scale)
        risk_factors = 0
        risk_factors += (data_with_risk['CreditRisk'] == 'Subprime').astype(int)
        risk_factors += (data_with_risk['DTIRisk'] == 'High').astype(int)
        risk_factors += (data_with_risk['DTIRisk'] == 'Very High').astype(int)
        risk_factors += (data_with_risk['CLTVRisk'] == 'High').astype(int)
        risk_factors += (data_with_risk['CLTVRisk'] == 'Very High').astype(int)
        
        data_with_risk['CombinedRisk'] = pd.cut(
            risk_factors, 
            bins=[-1, 0, 1, 2, 5], 
            labels=['Low', 'Moderate', 'High', 'Very High'],
            include_lowest=True
        )
        
        print("✅ Created risk categories: Credit, DTI, CLTV, and Combined")
        return data_with_risk
    
    def merge_datasets(self, orig_data: pd.DataFrame, perf_data: pd.DataFrame) -> pd.DataFrame:
        """Merge origination and performance datasets"""
        print("🔗 Merging origination and performance data...")
        
        # Merge on LoanSeqNumber
        merged = pd.merge(
            orig_data, 
            perf_data, 
            on='LoanSeqNumber', 
            how='inner',  # Only keep loans with both origination and performance data
            suffixes=('_orig', '_perf')
        )
        
        print(f"✅ Merged datasets: {len(merged):,} loans with complete data")
        return merged
    
    def create_target_variable(self, data: pd.DataFrame, latest_perf_data: pd.DataFrame) -> pd.DataFrame:
        """Create binary target variable using latest performance data"""
        print("🎯 Creating target variable using latest performance data...")
        
        data_with_target = data.copy()
        
        # Merge with latest performance data to get current DefaultStatus
        latest_default_status = latest_perf_data[['LoanSeqNumber', 'DefaultStatus']].copy()
        latest_default_status = latest_default_status.rename(columns={'DefaultStatus': 'DefaultStatus_current'})
        data_with_target = data_with_target.merge(
            latest_default_status, 
            on='LoanSeqNumber', 
            how='left'
        )
        
        # Handle DefaultStatus column - convert to numeric first
        # DefaultStatus: 0=Current, 1=30-59 days, 2=60-89 days, ..., RA=REO Acquisition
        def parse_default_status(status):
            if pd.isna(status):
                return 0
            if isinstance(status, str):
                if status == 'RA':  # REO Acquisition
                    return 10  # High value for REO
                try:
                    return int(status)
                except ValueError:
                    return 0
            return int(status)
        
        # Convert current DefaultStatus to numeric for target
        data_with_target['DefaultStatus_current_numeric'] = data_with_target['DefaultStatus_current'].apply(parse_default_status)
        
        # Create binary default indicator using current status
        data_with_target['Default'] = (
            data_with_target['DefaultStatus_current_numeric'] >= self.config.default_threshold
        ).astype(int)
        
        # Print default rate and status distribution
        default_rate = data_with_target['Default'].mean()
        status_dist = data_with_target['DefaultStatus_current'].value_counts()
        print(f"✅ Created target variable: {default_rate:.1%} default rate")
        print(f"📊 Current DefaultStatus distribution:")
        for status, count in status_dist.head(10).items():
            print(f"   {status}: {count:,} ({count/len(data_with_target):.1%})")
        
        return data_with_target
    
    def load_and_preprocess(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Main method to load and preprocess all data
        
        Returns:
            orig_data: Preprocessed origination data
            perf_data: Preprocessed performance data  
            merged_data: Merged dataset with target variable
        """
        print("🚀 Starting data loading and preprocessing...")
        print("="*60)
        
        # Load schemas
        self.load_schemas()
        
        # Load raw data
        orig_raw = self.load_origination_data()
        perf_raw = self.load_performance_data()
        
        # Get historical performance records (6 months ago) for features
        perf_historical = self.get_historical_performance(perf_raw, months_back=6)
        
        # Get latest performance records for target variable
        perf_latest = self.get_latest_performance(perf_raw)
        
        # Handle missing values
        self.orig_data = self.handle_missing_values(orig_raw, 'origination')
        self.perf_data = self.handle_missing_values(perf_historical, 'performance')
        
        # Create risk categories for origination data
        self.orig_data = self.create_risk_categories(self.orig_data)
        
        # Merge datasets (using historical performance for features)
        self.merged_data = self.merge_datasets(self.orig_data, self.perf_data)
        
        # Create target variable using latest performance data
        self.merged_data = self.create_target_variable(self.merged_data, perf_latest)
        
        print("\n✅ Data loading and preprocessing complete!")
        print(f"📊 Final dataset: {len(self.merged_data):,} loans")
        print(f"🎯 Default rate: {self.merged_data['Default'].mean():.1%}")
        
        return self.orig_data, self.perf_data, self.merged_data

def main():
    """Test the data loader"""
    config = DataConfig()
    loader = EMBSDataLoader(config)
    
    orig_data, perf_data, merged_data = loader.load_and_preprocess()
    
    print("\n📋 Dataset Summary:")
    print(f"Origination data shape: {orig_data.shape}")
    print(f"Performance data shape: {perf_data.shape}")
    print(f"Merged data shape: {merged_data.shape}")
    print(f"Default rate: {merged_data['Default'].mean():.1%}")

if __name__ == "__main__":
    main()
