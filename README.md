# Credit-Card-Fraud-Detection

## 📘 Project Overview
This project focuses on detecting fraudulent credit card transactions using machine learning. The goal was to compare various classification models and identify which performs best in detecting rare fraudulent cases.

### 🎯 Objective
To build and evaluate machine learning models that can classify transactions as **fraudulent** or **non-fraudulent** while addressing severe class imbalance.

---

## 🧠 Dataset
**Source:** [Credit Card Transactions Dataset](https://raw.githubusercontent.com/ArchanaInsights/Datasets/refs/heads/main/credit_card_transactions.csv)  
**Total Records:** 5,000 transactions  
**Target Column:** `is_fraud`  
**Challenge:** Highly imbalanced dataset — very few fraud cases compared to non-fraud.

---

## ⚙️ Methodology
1. **Data Preprocessing**
   - Handled missing values and outliers
   - Encoded categorical features
   - Scaled numerical variables for consistency

2. **Model Building**
   - Trained and compared multiple classifiers:
     - Logistic Regression  
     - GaussianNB  
     - Decision Tree  
     - Random Forest  
     - KNN  
     - SVC  

3. **Model Evaluation**
   - Evaluated models using:
     - Accuracy  
     - Precision  
     - Recall  
     - F1-Score  
     - Confusion Matrix  

---

## 🧩 Model Evaluation Summary

### ✅ Best Model: `DecisionTreeClassifier`

#### **Training Performance**
| Metric | Score |
|:-------|:------|
| Accuracy | 0.612 |
| Precision | 0.938 |
| Recall | 0.240 |
| F1-Score | 0.383 |

#### **Testing Performance**
| Metric | Score |
|:-------|:------|
| Accuracy | 0.713 |
| Precision | 0.200 |
| Recall | 0.026 |
| F1-Score | 0.047 |

#### **Confusion Matrix (Testing)**
|  | Predicted Non-Fraud | Predicted Fraud |
|:--|:--------------------|:----------------|
| **Actual Non-Fraud** | 706 (TN) | 28 (FP) |
| **Actual Fraud** | 259 (FN) | 7 (TP) |

---

## 📊 Insights
- The Decision Tree achieved the **highest accuracy (71.3%)**, mainly by predicting the majority (non-fraud) class correctly.
- However, **recall dropped sharply** from 24% (training) to 2.6% (testing), revealing **poor fraud detection capability**.
- The model is **overfitted** and **biased** toward non-fraud transactions.

---

## 🚀 Future Improvements
- Implement **SMOTE** or **undersampling** techniques to balance the dataset.  
- Use **ensemble models** (e.g., XGBoost, Random Forest with class weighting).  
- Perform **feature engineering** to improve minority class signals.  
- Apply **cross-validation** for more stable model evaluation.  

---

## 🧾 Conclusion
While the Decision Tree classifier showed the best accuracy, it failed to effectively detect fraud due to severe class imbalance.  
Future enhancements should focus on improving recall and generalization to ensure better fraud detection in real-world scenarios.



