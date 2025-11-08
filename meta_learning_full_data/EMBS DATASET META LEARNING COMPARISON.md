🚀 EMBS DATASET META LEARNING COMPARISON
================================================================================
Comparing Logistic Regression vs Meta Neural Networks
Using full EMBS dataset with origination and performance data
================================================================================

📊 STEP 1: LOADING AND PREPROCESSING DATA
--------------------------------------------------
🚀 Starting data loading and preprocessing...
============================================================
📋 Loading data schemas...
✅ Loaded schemas: 32 origination columns, 32 performance columns
📊 Loading origination data...
✅ Loaded 50,000 origination records
📊 Loading performance data...
✅ Loaded 969,328 performance records
🔄 Getting performance records from 6 months ago...
✅ Got historical performance for 50,000 loans (FAST!)
🔄 Getting latest performance records for target variable...
✅ Got latest performance for 50,000 loans
🔧 Handling missing values for origination data...
✅ Handled missing values for origination data
🔧 Handling missing values for performance data...
✅ Handled missing values for performance data
🎯 Creating risk categories...
✅ Created risk categories: Credit, DTI, CLTV, and Combined
🔗 Merging origination and performance data...
✅ Merged datasets: 50,000 loans with complete data
🎯 Creating target variable using latest performance data...
✅ Created target variable: 0.9% default rate
📊 Current DefaultStatus distribution:
   0: 49,110 (98.2%)
   1: 429 (0.9%)
   2: 120 (0.2%)
   5: 61 (0.1%)
   3: 55 (0.1%)
   4: 50 (0.1%)
   6: 37 (0.1%)
   8: 25 (0.1%)
   7: 23 (0.0%)
   9: 22 (0.0%)

✅ Data loading and preprocessing complete!
📊 Final dataset: 50,000 loans
🎯 Default rate: 0.9%

📋 Dataset Summary:
Total loans: 50,000
Default rate: 0.9%
Origination features: 36
Performance features: 32

🔧 STEP 2: FEATURE ENGINEERING
--------------------------------------------------
💰 Creating business features...
✅ Created business features
Training set: 40,000 loans
Test set: 10,000 loans
Training default rate: 0.9%
Test default rate: 0.9%
🔄 Fitting and transforming features...
🔧 Creating engineered features...
✅ Created engineered features
🔧 Creating engineered features...
✅ Created engineered features
🔧 Creating preprocessing pipelines...
✅ Created preprocessing pipelines
✅ Transformed features:
   Origination: 161 features
   Performance: 25 features

✅ Feature engineering complete:
Origination features: 161
Performance features: 25

🧠 STEP 3: TRAINING MODELS
--------------------------------------------------

📊 Training Logistic Regression...
✅ Logistic Regression trained in 1.83 seconds

============================================================
EVALUATING LOGISTIC REGRESSION
============================================================

Logistic Regression — Test @ threshold=0.50
-------------------------------------------
Accuracy:  0.8089 | Precision: 0.0290 | Recall: 0.6087 | F1: 0.0554
Confusion  TN=8,033  FP=1,875  FN=36  TP=56

Logistic Regression — Business-optimal threshold
------------------------------------------------
Threshold: 0.9500 | Profit: $158,211,290
Revenue:   $159,916,250 | Loss: $1,704,960
Accuracy:  0.9859 | Precision: 0.2118 | Recall: 0.1957 | F1: 0.2034
Confusion  TN=9,841  FP=67  FN=74  TP=18

ROC AUC: 0.7958

🧠 Training Simple Meta Neural Network...
2025-10-10 11:38:33.899076: I metal_plugin/src/device/metal_device.cc:1154] Metal device set to: Apple M2
2025-10-10 11:38:33.899114: I metal_plugin/src/device/metal_device.cc:296] systemMemory: 16.00 GB
2025-10-10 11:38:33.899121: I metal_plugin/src/device/metal_device.cc:313] maxCacheSize: 5.33 GB
2025-10-10 11:38:33.899322: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
2025-10-10 11:38:33.899343: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
2025-10-10 11:38:35.008704: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:117] Plugin optimizer for device_type GPU is enabled.
✅ Simple Meta NN trained in 168.02 seconds

============================================================
EVALUATING SIMPLE META NN
============================================================

Simple Meta NN — Test @ threshold=0.50
--------------------------------------
Accuracy:  0.9905 | Precision: 0.2000 | Recall: 0.0109 | F1: 0.0206
Confusion  TN=9,904  FP=4  FN=91  TP=1

Simple Meta NN — Business-optimal threshold
-------------------------------------------
Threshold: 0.9000 | Profit: $158,908,360
Revenue:   $161,005,000 | Loss: $2,096,640
Accuracy:  0.9909 | Precision: 1.0000 | Recall: 0.0109 | F1: 0.0215
Confusion  TN=9,908  FP=0  FN=91  TP=1

ROC AUC: 0.7940

🚀 Training Enhanced Meta Neural Network...
✅ Enhanced Meta NN trained in 700.54 seconds

============================================================
EVALUATING ENHANCED META NN
============================================================

Enhanced Meta NN — Test @ threshold=0.50
----------------------------------------
Accuracy:  0.9908 | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0000
Confusion  TN=9,908  FP=0  FN=92  TP=0

Enhanced Meta NN — Business-optimal threshold
---------------------------------------------
Threshold: 0.0500 | Profit: $158,885,320
Revenue:   $161,005,000 | Loss: $2,119,680
Accuracy:  0.9908 | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0000
Confusion  TN=9,908  FP=0  FN=92  TP=0

ROC AUC: 0.7051

📊 STEP 4: MODEL COMPARISON
--------------------------------------------------

================================================================================
MODEL COMPARISON AND RANKING
================================================================================
Rank Model                     Profit ($)      Threshold  AUC      Recall   F1      
----------------------------------------------------------------------------------------------------
1    Simple Meta NN            $158,908,360    0.9000     0.7940   0.0109   0.0215  
2    Enhanced Meta NN          $158,885,320    0.0500     0.7051   0.0000   0.0000  
3    Logistic Regression       $158,211,290    0.9500     0.7958   0.1957   0.2034  

🔍 KEY INSIGHTS:
================================================================================
🥇 Best Model: Simple Meta NN
   💰 Profit: $158,908,360
   📈 Recall: 1.1%
   🎯 AUC: 0.7940

🥉 Worst Model: Logistic Regression
   💰 Profit: $158,211,290
   📈 Recall: 19.6%
   🎯 AUC: 0.7958

🚀 Best vs Worst:
   💰 Profit improvement: $697,070 (0.44%)
   📈 Recall improvement: -18.5%

💰 BUSINESS IMPACT ANALYSIS
================================================================================
Best Model: Simple Meta NN
Worst Model: Logistic Regression

💰 Annual Profit Improvement: $697,070
📈 Improvement Percentage: 0.44%
🎯 ROI: High (Meta NN provides measurable business value)

⚖️  Risk Analysis:
Best Model Recall: 1.1%
Worst Model Recall: 19.6%
Recall Difference: -18.5%
⚠️  Best model has lower recall but higher profit (optimized for business)

🎉 FINAL RESULTS SUMMARY
================================================================================

⏱️  Training Times:
Logistic Regression: 1.83 seconds
Simple Meta NN: 168.02 seconds
Enhanced Meta NN: 700.54 seconds

💰 Best Model: Simple Meta NN
Annual Profit Improvement: $697,070
Improvement Percentage: 0.44%

📈 Key Insights:
• Dataset size: 50,000 loans
• Default rate: 0.9%
• Feature engineering: 161 origination + 25 performance features
• Best model profit: $158,908,360
• Best model recall: 1.1%

💾 SAVING RESULTS
--------------------------------------------------
✅ Saved model comparison results to 'model_comparison_results.csv'
✅ Saved model predictions to 'model_predictions.csv'

🎯 COMPARISON COMPLETE!
================================================================================

✅ All done! Check the generated CSV files for detailed results.