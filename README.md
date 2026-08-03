# Machine Learning-based Heart Disease Prediction System

## Overview
A machine learning pipeline for predicting heart disease from patient 
clinical data. The project systematically compares 12 classification 
algorithms and 3 feature selection techniques to identify the most 
accurate and efficient model configuration.

## Dataset
- UCI Heart Disease dataset — 1,025 records, 13 clinical features 
  (age, chest pain type, resting blood pressure, cholesterol, max heart rate, etc.)

## Pipeline
1. **Preprocessing** — missing value imputation (mean/median/KNN), 
   categorical encoding, MinMax feature scaling
2. **Feature selection** — compared 3 techniques: Recursive Feature 
   Elimination (RFE), Principal Component Analysis (PCA), and 
   Univariate Feature Selection
3. **Modeling** — trained and evaluated 12 classifiers (Logistic Regression, 
   SVM, Random Forest, AdaBoost, Decision Tree, Naive Bayes, KNN, LDA, 
   Extra Trees, Gradient Boosting, Bagging, and a custom Voting Ensemble)
4. **Tuning** — hyperparameter optimization via GridSearchCV with 
   5-fold stratified cross-validation

## Results
- **Best model: Decision Tree (CART)** — 100% cross-validated accuracy 
  (tuned via GridSearchCV), outperforming all 11 other classifiers tested
- Feature selection (RFE / Univariate) reduced dimensionality from 
  13 → 10 features while maintaining 99.6–100% accuracy
- Built a custom Voting Ensemble Classifier combining Random Forest, 
  Extra Trees, Gradient Boosting, SVM, and Logistic Regression for 
  improved robustness

## Additional experiments
- Compared imputation strategies (mean, median, KNN) for handling missing data
- Compared categorical encoding strategies and their effect on model accuracy

## Tech stack
Python, scikit-learn, pandas, NumPy, matplotlib

## How to run
1. Install dependencies: `pip install pandas numpy scikit-learn matplotlib`
2. Place the dataset CSV in the expected data folder (see notebook for path)
3. Run `miniproject.ipynb` cell by cell, or execute the full pipeline via the `__main__` block
