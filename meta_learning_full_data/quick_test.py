#!/usr/bin/env python3
"""
Quick test script for data loader
"""
import sys
import time

def test_data_loader():
    try:
        print("🧪 Testing Data Loader...")
        start_time = time.time()
        
        from data_loader import DataConfig, EMBSDataLoader
        
        config = DataConfig()
        loader = EMBSDataLoader(config)
        
        print("✅ Imports successful")
        
        # Test schema loading
        print("📋 Loading schemas...")
        loader.load_schemas()
        print("✅ Schemas loaded")
        
        # Test origination data (smaller file)
        print("📊 Loading origination data...")
        orig_data = loader.load_origination_data()
        print(f"✅ Origination data: {orig_data.shape}")
        
        # Test performance data loading (this might be slow)
        print("📊 Loading performance data...")
        perf_data = loader.load_performance_data()
        print(f"✅ Performance data: {perf_data.shape}")
        
        # Test historical performance
        print("🔄 Testing historical performance...")
        perf_historical = loader.get_historical_performance(perf_data, months_back=6)
        print(f"✅ Historical performance: {perf_historical.shape}")
        
        # Test latest performance
        print("🔄 Testing latest performance...")
        perf_latest = loader.get_latest_performance(perf_data)
        print(f"✅ Latest performance: {perf_latest.shape}")
        
        # Check distributions
        print("📊 DefaultStatus distributions:")
        print("Historical:", perf_historical['DefaultStatus'].value_counts().head(3).to_dict())
        print("Latest:", perf_latest['DefaultStatus'].value_counts().head(3).to_dict())
        
        elapsed = time.time() - start_time
        print(f"✅ All tests passed in {elapsed:.2f} seconds")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_loader()
    sys.exit(0 if success else 1)



