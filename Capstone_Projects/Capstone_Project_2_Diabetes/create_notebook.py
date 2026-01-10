import json
import os

notebook_filename = 'Diabetes_Analysis.ipynb'
dataset_filename = 'diabetes.csv'

cells = []

def add_md(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + '\n' for line in source]
    })

def add_code(source):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in source]
    })

# Header
add_md(["# Capstone Project 2: Pima Indians Diabetes Analysis", "## Comprehensive Analysis covering Modules 1-8"])

# Dependencies
add_code(["!pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn minisom"])

# Imports
add_code([
    "import pandas as pd",
    "import numpy as np",
    "import matplotlib.pyplot as plt",
    "import seaborn as sns",
    "from sklearn.impute import SimpleImputer, KNNImputer",
    "from sklearn.experimental import enable_iterative_imputer",
    "from sklearn.impute import IterativeImputer",
    "from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler",
    "from sklearn.preprocessing import PowerTransformer",
    "from sklearn.model_selection import train_test_split",
    "from sklearn.tree import DecisionTreeClassifier, export_text",
    "from sklearn.naive_bayes import GaussianNB",
    "from sklearn.neural_network import MLPClassifier",
    "from sklearn.linear_model import LinearRegression",
    "from sklearn.cluster import KMeans, DBSCAN, MeanShift",
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix, silhouette_score",
    "from imblearn.over_sampling import SMOTE, RandomOverSampler",
    "from imblearn.under_sampling import RandomUnderSampler",
    "from minisom import MiniSom",
    "import warnings",
    "warnings.filterwarnings('ignore')"
])

# Module 1: Data Cleaning
add_md(["## Module 1: Data Cleaning", "Apply Feature Imputing (Simple, KNN, Iterative). Ensure missing values exist."])
add_code([
    "# Load Dataset",
    f"df = pd.read_csv('{dataset_filename}', header=None)",
    "columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']",
    "df.columns = columns",
    "print('Original Data Shape:', df.shape)",
    "df.head()"
])
add_code([
    "# Replace 0 with NaN for columns where 0 is invalid",
    "cols_missing_vals = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']",
    "df[cols_missing_vals] = df[cols_missing_vals].replace(0, np.nan)",
    "print('\\nMissing values count:')",
    "print(df.isnull().sum())"
])
add_code([
    "# 1. Simple Imputer (Mean)",
    "simple_imputer = SimpleImputer(strategy='mean')",
    "df_simple = pd.DataFrame(simple_imputer.fit_transform(df), columns=df.columns)",
    "",
    "# 2. KNN Imputer",
    "knn_imputer = KNNImputer(n_neighbors=5)",
    "df_knn = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)",
    "",
    "# 3. Iterative Imputer",
    "iter_imputer = IterativeImputer(max_iter=10, random_state=0)",
    "df_iter = pd.DataFrame(iter_imputer.fit_transform(df), columns=df.columns)",
    "",
    "print('\\nImputation completed. Using KNN Imputed data for further analysis (common practice).')",
    "df_clean = df_knn.copy()",
    "df_clean['Outcome'] = df_clean['Outcome'].round().astype(int) # Outcome should be int"
])

# Module 2: Symmetry Check
add_md(["## Module 2: Symmetry Check", "Compute Mean, Median, Skewness. Classify Features."])
add_code([
    "stats = pd.DataFrame()",
    "stats['Mean'] = df_clean.mean()",
    "stats['Median'] = df_clean.median()",
    "stats['Skewness'] = df_clean.skew()",
    "",
    "def classify_skew(skew):",
    "    if -0.5 <= skew <= 0.5:",
    "        return 'Symmetric'",
    "    elif skew > 0.5:",
    "        return 'Right-Skewed'",
    "    else:",
    "        return 'Left-Skewed'",
    "",
    "stats['Skew Analysis'] = stats['Skewness'].apply(classify_skew)",
    "stats"
])

# Module 3: Data Transformations
add_md(["## Module 3: Data Transformations", "Apply Scalers and Transformations to fix skewness."])
add_code([
    "X = df_clean.drop('Outcome', axis=1)",
    "y = df_clean['Outcome']",
    "",
    "# Scalers",
    "scalers = {",
    "    'Standard': StandardScaler(),",
    "    'MinMax': MinMaxScaler(),",
    "    'Robust': RobustScaler(),",
    "    'MaxAbs': MaxAbsScaler()",
    "}",
    "",
    "for name, scaler in scalers.items():",
    "    print(f'Applying {name} Scaler...')",
    "    scaler.fit_transform(X)",
    "",
    "# Transformations (focusing on skewed features)",
    "# Using Yeo-Johnson as it handles zero and negative values better than Box-Cox",
    "pt = PowerTransformer(method='yeo-johnson')",
    "X_transformed = pd.DataFrame(pt.fit_transform(X), columns=X.columns)",
    "",
    "print('\\nSkewness after Yeo-Johnson Transformation:')",
    "print(X_transformed.skew())"
])

# Module 4: Univariate Data Visualization
add_md(["## Module 4: Univariate Data Visualization", "Bar plot, Histogram, Box plot."])
add_code([
    "plt.figure(figsize=(10, 6))",
    "sns.countplot(x='Outcome', data=df_clean)",
    "plt.title('Distribution of Outcome (Bar Plot)')",
    "plt.show()"
])
add_code([
    "features = X.columns",
    "plt.figure(figsize=(15, 10))",
    "for i, feature in enumerate(features):",
    "    plt.subplot(3, 3, i+1)",
    "    sns.histplot(df_clean[feature], kde=True)",
    "    plt.title(f'Histogram of {feature}')",
    "plt.tight_layout()",
    "plt.show()"
])
add_code([
    "plt.figure(figsize=(15, 10))",
    "for i, feature in enumerate(features):",
    "    plt.subplot(3, 3, i+1)",
    "    sns.boxplot(x=df_clean[feature])",
    "    plt.title(f'Boxplot of {feature}')",
    "plt.tight_layout()",
    "plt.show()"
])

# Module 5: Balancing Dataset
add_md(["## Module 5: Balancing Dataset", "Undersampling, Oversampling, SMOTE. Entropy Analysis."])
add_code([
    "def calculate_entropy(y):",
    "    elements, counts = np.unique(y, return_counts=True)",
    "    probabilities = counts / len(y)",
    "    entropy = -np.sum(probabilities * np.log2(probabilities))",
    "    return entropy",
    "",
    "# 1. Imbalanced (Original)",
    "entropy_imbalanced = calculate_entropy(y)",
    "print(f'Entropy of Original (Imbalanced) Dataset: {entropy_imbalanced:.4f}')",
    "",
    "# 2. Undersampling",
    "rus = RandomUnderSampler(random_state=42)",
    "X_res_under, y_res_under = rus.fit_resample(X_transformed, y)",
    "entropy_under = calculate_entropy(y_res_under)",
    "print(f'Entropy of Undersampled Dataset: {entropy_under:.4f}')",
    "",
    "# 3. Oversampling (SMOTE)",
    "smote = SMOTE(random_state=42)",
    "X_res_smote, y_res_smote = smote.fit_resample(X_transformed, y)",
    "entropy_smote = calculate_entropy(y_res_smote)",
    "print(f'Entropy of SMOTE Balanced Dataset: {entropy_smote:.4f}')",
    "",
    "# Analysis note",
    "print('\\nAnalysis: Balanced datasets (Entropy ~ 1.0) generally reduce bias towards the majority class but may increase variance. Imbalanced datasets have lower entropy and high bias towards majority.')"
])

# Module 6: Supervised Learning
add_md(["## Module 6: Supervised Learning", "ID3, Bayes, MLP, Linear Regression."])
add_code([
    "# Using SMOTE balanced data for better performance",
    "X_train, X_test, y_train, y_test = train_test_split(X_res_smote, y_res_smote, test_size=0.2, random_state=42)",
    "",
    "models = {}",
    "",
    "# a) ID3 Decision Tree (using entropy for Information Gain)",
    "dt = DecisionTreeClassifier(criterion='entropy', random_state=42)",
    "dt.fit(X_train, y_train)",
    "models['Decision Tree (ID3)'] = dt",
    "print('Decision Tree Rules (First 500 chars):')",
    "print(export_text(dt, feature_names=list(X.columns))[:500])",
    "",
    "# b) Bayesian Classifier",
    "nb = GaussianNB()",
    "nb.fit(X_train, y_train)",
    "models['Naive Bayes'] = nb",
    "",
    "# c) MLP Custom Hypertuning",
    "mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500, random_state=42)",
    "mlp.fit(X_train, y_train)",
    "models['MLP Classifier'] = mlp",
    "",
    "# d) Multivariate Linear Regression (for prediction, rounding output for classification ref)",
    "lr = LinearRegression()",
    "lr.fit(X_train, y_train)",
    "models['Linear Regression'] = lr"
])

# Module 7: Unsupervised Learning
add_md(["## Module 7: Unsupervised Learning", "K-Means, SOM, DBSCAN, Mean Shift."])
add_code([
    "# Using X_transformed (Standardized/Normalized)",
    "clustering_results = {}",
    "",
    "# a) K-Means",
    "kmeans = KMeans(n_clusters=2, random_state=42)",
    "labels_kmeans = kmeans.fit_predict(X_transformed)",
    "clustering_results['KMeans'] = labels_kmeans",
    "",
    "# b) SOM Clustering (using MiniSom)",
    "# 10x10 map",
    "som = MiniSom(x=10, y=10, input_len=X_transformed.shape[1], sigma=1.0, learning_rate=0.5)",
    "som.random_weights_init(X_transformed.values)",
    "som.train_random(X_transformed.values, 100)",
    "# Assigning each point to its BMU",
    "labels_som = [som.winner(x) for x in X_transformed.values]",
    "# SOM labels are tuples (coords), plotting needs conversion or just analysis. Storing as str for now.",
    "clustering_results['SOM'] = [str(x) for x in labels_som]",
    "",
    "# c) DBSCAN",
    "dbscan = DBSCAN(eps=0.5, min_samples=5)",
    "labels_dbscan = dbscan.fit_predict(X_transformed)",
    "clustering_results['DBSCAN'] = labels_dbscan",
    "",
    "# d) Mean Shift",
    "meanshift = MeanShift()",
    "labels_meanshift = meanshift.fit_predict(X_transformed)",
    "clustering_results['MeanShift'] = labels_meanshift"
])

# Module 8: Performance Metrics
add_md(["## Module 8: Performance Metrics", "Sensitivity, Specificity, Accuracy, Precision, Recall, F1, ROC, Silhouette."])
add_code([
    "for name, model in models.items():",
    "    print(f'--- {name} ---')",
    "    if name == 'Linear Regression':",
    "        y_pred_raw = model.predict(X_test)",
    "        y_pred = np.round(y_pred_raw).astype(int)",
    "        y_pred = np.clip(y_pred, 0, 1)",
    "    else:",
    "        y_pred = model.predict(X_test)",
    "    ",
    "    # Confusion Matrix",
    "    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()",
    "    ",
    "    # Metrics",
    "    accuracy = accuracy_score(y_test, y_pred)",
    "    precision = precision_score(y_test, y_pred)",
    "    recall = recall_score(y_test, y_pred) # Sensitivity",
    "    specificity = tn / (tn + fp)",
    "    f1 = f1_score(y_test, y_pred)",
    "    ",
    "    print(f'Accuracy: {accuracy:.4f}')",
    "    print(f'Sensitivity (Recall): {recall:.4f}')",
    "    print(f'Specificity: {specificity:.4f}')",
    "    print(f'Precision: {precision:.4f}')",
    "    print(f'F1 Score: {f1:.4f}')",
    "    ",
    "    if hasattr(model, 'predict_proba'):",
    "        probs = model.predict_proba(X_test)[:, 1]",
    "        auc = roc_auc_score(y_test, probs)",
    "        print(f'ROC AUC: {auc:.4f}')",
    "    print('\\n')"
])
add_code([
    "# Silhouette Score for Clustering",
    "print('--- Clustering Silhouette Scores ---')",
    "for name, labels in clustering_results.items():",
    "    # Silhouette score requires >1 unique label and valid labels",
    "    unique_labels = len(set(labels))",
    "    if unique_labels > 1 and unique_labels < len(X_transformed):",
    "        # For SOM, labels are strings of coordinates, need to encode or skip",
    "        if name == 'SOM':",
    "             # Flatten coords for silhouette if needed, or skip",
    "             pass ",
    "        else:",
    "             sil = silhouette_score(X_transformed, labels)",
    "             print(f'{name}: {sil:.4f}')",
    "    else:",
    "        print(f'{name}: Not applicable (1 cluster or noise)')"
])

# Create the notebook
notebook_json = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(notebook_filename, 'w', encoding='utf-8') as f:
    json.dump(notebook_json, f, indent=1)

print(f"Notebook {notebook_filename} created successfully.")
