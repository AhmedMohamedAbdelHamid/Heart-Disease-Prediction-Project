# Heart-Disease-Prediction-Project
This project explores **machine learning models** for predicting heart disease using the UCI Heart Disease dataset.  
It includes **data preprocessing, feature engineering, dimensionality reduction, supervised & unsupervised learning, and hyperparameter tuning**.

---
## Business-Wise Model Insights

### Logistic Regression (baseline model)
  **Accuracy:** ~99%  
- **AUC:** ~0.98 (excellent separability of risky vs non-risky patients)  
- **Precision (Heart Disease = 1):** ~0.88–1.00  
- **Recall (Heart Disease = 1):** ~0.34–0.40 (many true cases missed)  
- **F1 Score (Heart Disease = 1):** ~0.51–0.55  

  **Business Meaning** 
- High AUC shows the model can rank patients well.  
- But low recall means the system **misses many at-risk patients**, which is unacceptable in a clinical setting.  
- **Recommendation:** Adjust thresholds, use ensemble models (e.g., RandomForest), and rebalance data to improve recall.  

---

## Workflow
### 1. Data Preprocessing
- Load multiple `.data` files
- Clean missing values, assign headers
- Save cleaned dataset (`data/heart_disease_clean.csv`)

### 2. PCA Analysis
- Dimensionality reduction to visualize variance explained
- Scree plot included

### 3. Feature Selection
- RandomForest importance ranking
- Recursive Feature Elimination (RFE)

### 4. Supervised Learning
- Logistic Regression, RandomForest, SVM, DecisionTree
- Performance metrics: Accuracy, Precision, Recall, F1, AUC

### 5. Unsupervised Learning
- KMeans with Elbow method (best k detection)
- Agglomerative Clustering (with subsampling for large data)

### 6. Hyperparameter Tuning
- GridSearchCV on RandomForest
- Final model saved as `final_model.pkl`

---

## Results
- RandomForest achieved the best performance after tuning  
- PCA helped reduce dimensionality without much accuracy loss  
- KMeans showed weak separation compared to supervised models  
