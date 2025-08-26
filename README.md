# Fraud Report Classification – Canadian Anti-Fraud Centre (Python)
#### Link to download dataset : https://open.canada.ca/data/en/dataset/6a09c998-cddb-4a22-beff-4dca67ab892f/resource/43c67af5-e598-4a9b-a484-fe1cb5d775b5

This project applies **machine learning and statistical analysis** to fraud report data from the Canadian Anti-Fraud Centre.  
The goal is to **predict fraud classifications** while handling real-world challenges such as missing values, outliers, collinearity, and imbalanced data.  

---

## 📊 Dataset
- Fraud-report dataset (anonymized categorical and numeric variables).  
- > 1,000 records, > 5 features (meeting project requirements).  
- Contains anonymized categorical variables (interpretability not critical; focus is on **predictive performance**).  

---

## 🧭 Project Steps

### 🔹 Data Preparation
1. Import dataset into Python (pandas).  
2. Display first and last 5 records.  
3. Identify variable data types.  
4. Handle missing values (`NaN`) and count them per variable.  
5. Detect and remove duplicate records.  

### 🔹 Exploratory Data Analysis
6. **Univariate analysis** – visualizations for numeric & categorical variables.  
7. **Outlier detection** – applied *Local Outlier Factor (LoF)*.  
8. **Bivariate analysis** –  
   - Chi-square tests (categorical vs categorical).  
   - Correlation analysis (numeric vs numeric).  
9. Checked for **collinearity and multi-collinearity** and corrected where necessary.  

### 🔹 Feature Engineering
10. Encoded categorical variables.  
11. Train/test split.  
12. Scaled data (standardization for models requiring scaling).  
13. Applied **PCA** and evaluated if components improved model preparation.  

### 🔹 Modeling (Classification Goal)
14. Built at least 3 models:  
   - Logistic Regression  
   - Random Forest  
   - XGBoost (with hyperparameter tuning via GridSearchCV)  
15. Checked and addressed **overfitting** using cross-validation and regularization.  

---

## 🧮 Evaluation

- **Classification metrics**:  
  - Accuracy  
  - Confusion Matrix  
  - Macro Precision & Recall  
  - F1-score  
- **K-Fold Cross Validation** applied to ensure robustness.  

---

## 📈 Findings & Results

- Logistic Regression offered baseline interpretability.  
- Random Forest and XGBoost provided **superior predictive performance**, especially with imbalanced data handling.  
- PCA reduced dimensionality but did not significantly improve predictive performance compared to tuned models.  
- Final recommendation: **XGBoost** with tuned hyperparameters delivered the best balance of accuracy and recall.  

---

## 📂 Repository Structure
