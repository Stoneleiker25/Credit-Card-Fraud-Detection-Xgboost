# Credit Card Fraud Detection with XGBoost

## Project Overview
This project builds a machine learning pipeline to detect credit card fraud using a highly imbalanced dataset (0.17% fraud). The entire solution was implemented locally in Python without cloud infrastructure. The goal was to identify as many fraud cases as possible while minimizing false positives.

## Objectives
- Detect rare fraud cases using a high-volume, imbalanced dataset
- Build an XGBoost model using Python and scikit-learn
- Tune prediction thresholds to prioritize recall
- Evaluate performance using precision and fraud recovery rate

## Dataset Summary
- Total transactions: 284,807  
- Fraudulent transactions: 492  
- Non-fraudulent transactions: 284,315  

## Tools and Libraries
- Python (Jupyter Notebook, Anaconda)
- XGBoost
- scikit-learn (StandardScaler, train_test_split, metrics)
- pandas, numpy

## Methodology
1. Dropped unnecessary column (‘Time’)
2. Scaled all features using `StandardScaler`
3. Split data into train/test sets and trained an XGBoost model
4. Ran predictions using multiple thresholds (0.5 → 0.1)
5. Retrained model on full dataset to maximize fraud detection
6. Evaluated performance using fraud counts and precision

## Results
- Actual frauds in dataset: 492  
- Predicted frauds at threshold 0.1: 506  
- Nearly all real frauds were successfully identified  
- Low false positive rate maintained  

## Key Takeaways
- Lowering the threshold to 0.1 dramatically improved recall
- Training on the full dataset helped the model learn rare fraud patterns
- XGBoost handled numeric, structured data efficiently
- Even without cloud tools, strong models can be built locally

## File Descriptions
- `credit_fraud_detection.ipynb`: Full notebook with code and results  
- `credit_fraud_results.pdf`: Output export of notebook  
- `Final_Summary.pdf`: Executive summary of the approach and findings  
- `Credit Card Fraud Detection (1).csv`: Original dataset (if shared)

## Next Steps
- Add precision/recall visualization
- Optional deployment using Flask or Streamlit  
- Extend to real-time fraud stream simulation

