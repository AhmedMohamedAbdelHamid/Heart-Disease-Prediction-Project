# Heart-Disease-Prediction-Project
This project explores **machine learning models** for predicting heart disease using the UCI Heart Disease dataset.  
It includes **data preprocessing, feature engineering, dimensionality reduction, supervised & unsupervised learning, and hyperparameter tuning**.

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
