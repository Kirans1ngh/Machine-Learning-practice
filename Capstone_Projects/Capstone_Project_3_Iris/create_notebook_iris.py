import json
import os

notebook_filename = 'Iris_Analysis.ipynb'
dataset_filename = 'iris.csv'

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
add_md(["# Capstone Project 3: Iris Dataset Analysis", "## Comprehensive Analysis covering Modules 1-8"])

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
    "from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, LabelEncoder",
    "from sklearn.preprocessing import PowerTransformer",
    "from sklearn.model_selection import train_test_split",
    "from sklearn.tree import DecisionTreeClassifier, export_text",
    "from sklearn.naive_bayes import GaussianNB",
    "from sklearn.neural_network import MLPClassifier",
    "from sklearn.linear_model import LinearRegression",
    "from sklearn.cluster import KMeans, DBSCAN, MeanShift",
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, silhouette_score, classification_report",
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
    "columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']",
    "df.columns = columns",
    "print('Original Data Shape:', df.shape)",
    "df.head()"
])
add_code([
    "# Introduce missing values randomly to demonstrate imputation",
    "np.random.seed(42)",
    "df_missing = df.copy()",
    "mask = np.random.rand(*df_missing.iloc[:, :-1].shape) < 0.1 # 10% missing",
    "df_missing.iloc[:, :-1] = df_missing.iloc[:, :-1].mask(mask)",
    "print('\\nMissing values count (Artificially Introduced):')",
    "print(df_missing.isnull().sum())"
])
add_code([
    "# Separate features and target",
    "X_missing = df_missing.drop('species', axis=1)",
    "y = df_missing['species']",
    "",
    "# 1. Simple Imputer (Mean)",
    "simple_imputer = SimpleImputer(strategy='mean')",
    "X_simple = pd.DataFrame(simple_imputer.fit_transform(X_missing), columns=X_missing.columns)",
    "",
    "# 2. KNN Imputer",
    "knn_imputer = KNNImputer(n_neighbors=5)",
    "X_knn = pd.DataFrame(knn_imputer.fit_transform(X_missing), columns=X_missing.columns)",
    "",
    "# 3. Iterative Imputer",
    "iter_imputer = IterativeImputer(max_iter=10, random_state=0)",
    "X_iter = pd.DataFrame(iter_imputer.fit_transform(X_missing), columns=X_missing.columns)",
    "",
    "print('\\nImputation completed. Using KNN Imputed data for further analysis.')",
    "df_clean = X_knn.copy()",
    "df_clean['species'] = y.values"
])

# Module 2: Symmetry Check
add_md(["## Module 2: Symmetry Check", "Compute Mean, Median, Skewness. Classify Features."])
add_code([
    "stats = pd.DataFrame()",
    "numeric_cols = df_clean.select_dtypes(include=np.number).columns",
    "stats['Mean'] = df_clean[numeric_cols].mean()",
    "stats['Median'] = df_clean[numeric_cols].median()",
    "stats['Skewness'] = df_clean[numeric_cols].skew()",
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
    "X = df_clean.drop('species', axis=1)",
    "# Encode target for modeling",
    "le = LabelEncoder()",
    "y_encoded = le.fit_transform(df_clean['species'])",
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
    "# Transformations (Yeo-Johnson)",
    "pt = PowerTransformer(method='yeo-johnson')",
    "X_transformed = pd.DataFrame(pt.fit_transform(X), columns=X.columns)",
    "",
    "print('\\nSkewness after Yeo-Johnson Transformation:')",
    "print(X_transformed.skew())"
])

# Module 4: Univariate Data Visualization
add_md(["## Module 4: Univariate Data Visualization", "Bar plot, Histogram, Box plot."])
add_code([
    "plt.figure(figsize=(8, 5))",
    "sns.countplot(x='species', data=df_clean)",
    "plt.title('Distribution of Species (Bar Plot)')",
    "plt.show()"
])
add_code([
    "features = X.columns",
    "plt.figure(figsize=(12, 10))",
    "for i, feature in enumerate(features):",
    "    plt.subplot(2, 2, i+1)",
    "    sns.histplot(df_clean[feature], kde=True)",
    "    plt.title(f'Histogram of {feature}')",
    "plt.tight_layout()",
    "plt.show()"
])
add_code([
    "plt.figure(figsize=(12, 10))",
    "for i, feature in enumerate(features):",
    "    plt.subplot(2, 2, i+1)",
    "    sns.boxplot(x=df_clean[feature])",
    "    plt.title(f'Boxplot of {feature}')",
    "plt.tight_layout()",
    "plt.show()"
])

# Module 5: Balancing Dataset
add_md(["## Module 5: Balancing Dataset", "Create artificial imbalance to demonstrate methods. Undersampling, Oversampling, SMOTE. Entropy Analysis."])
add_code([
    "def calculate_entropy(y):",
    "    elements, counts = np.unique(y, return_counts=True)",
    "    probabilities = counts / len(y)",
    "    entropy = -np.sum(probabilities * np.log2(probabilities))",
    "    return entropy",
    "",
    "# Create Artificial Imbalance (remove 90% of class 0)",
    "mask = (y_encoded != 0) | (np.random.rand(len(y_encoded)) < 0.1)",
    "X_imbal = X_transformed[mask]",
    "y_imbal = y_encoded[mask]",
    "print('Imbalanced Class Counts:', np.bincount(y_imbal))",
    "",
    "# 1. Imbalanced",
    "entropy_imbalanced = calculate_entropy(y_imbal)",
    "print(f'Entropy of Imbalanced Dataset: {entropy_imbalanced:.4f}')",
    "",
    "# 2. Undersampling",
    "# Note: With 3 classes, it undersamples the majority classes to match the minority",
    "rus = RandomUnderSampler(random_state=42)",
    "X_res_under, y_res_under = rus.fit_resample(X_imbal, y_imbal)",
    "entropy_under = calculate_entropy(y_res_under)",
    "print(f'Entropy of Undersampled Dataset: {entropy_under:.4f}')",
    "",
    "# 3. Oversampling (SMOTE)",
    "# SMOTE will oversample minority classes",
    "smote = SMOTE(random_state=42)",
    "X_res_smote, y_res_smote = smote.fit_resample(X_imbal, y_imbal)",
    "entropy_smote = calculate_entropy(y_res_smote)",
    "print(f'Entropy of SMOTE Balanced Dataset: {entropy_smote:.4f}')",
    "print('Balanced Class Counts:', np.bincount(y_res_smote))"
])

# Module 6: Supervised Learning
add_md(["## Module 6: Supervised Learning", "ID3, Bayes, MLP, Linear Regression."])
add_code([
    "# Using original balanced transformed data for standard classification task",
    "X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_encoded, test_size=0.2, random_state=42)",
    "",
    "models = {}",
    "",
    "# a) ID3 Decision Tree (Information Gain)",
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
    "# d) Multivariate Linear Regression",
    "# Predicting continuous value close to class ID (0, 1, 2)",
    "lr = LinearRegression()",
    "lr.fit(X_train, y_train)",
    "models['Linear Regression'] = lr"
])

# Module 7: Unsupervised Learning
add_md(["## Module 7: Unsupervised Learning", "K-Means, SOM, DBSCAN, Mean Shift."])
add_code([
    "clustering_results = {}",
    "",
    "# a) K-Means (3 clusters)",
    "kmeans = KMeans(n_clusters=3, random_state=42)",
    "labels_kmeans = kmeans.fit_predict(X_transformed)",
    "clustering_results['KMeans'] = labels_kmeans",
    "",
    "# b) SOM Clustering",
    "som = MiniSom(x=10, y=10, input_len=X_transformed.shape[1], sigma=1.0, learning_rate=0.5)",
    "som.random_weights_init(X_transformed.values)",
    "som.train_random(X_transformed.values, 100)",
    "labels_som = [som.winner(x) for x in X_transformed.values]",
    "clustering_results['SOM'] = [str(x) for x in labels_som]",
    "",
    "# c) DBSCAN",
    "dbscan = DBSCAN(eps=0.8, min_samples=5)",
    "labels_dbscan = dbscan.fit_predict(X_transformed)",
    "clustering_results['DBSCAN'] = labels_dbscan",
    "",
    "# d) Mean Shift",
    "meanshift = MeanShift()",
    "labels_meanshift = meanshift.fit_predict(X_transformed)",
    "clustering_results['MeanShift'] = labels_meanshift"
])

# Module 8: Performance Metrics
add_md(["## Module 8: Performance Metrics", "Acc, Prec, Recall, F1, ROC, Silhouette."])
add_code([
    "for name, model in models.items():",
    "    print(f'--- {name} ---')",
    "    if name == 'Linear Regression':",
    "        y_pred_raw = model.predict(X_test)",
    "        y_pred = np.rint(y_pred_raw).astype(int)",
    "        y_pred = np.clip(y_pred, 0, 2)",
    "    else:",
    "        y_pred = model.predict(X_test)",
    "    ",
    "    print(classification_report(y_test, y_pred, target_names=le.classes_))",
    "    ",
    "    # Confusion Matrix",
    "    cm = confusion_matrix(y_test, y_pred)",
    "    print('Confusion Matrix:\\n', cm)",
    "    ",
    "    # ROC AUC (One-vs-Rest for Multi-class)",
    "    if hasattr(model, 'predict_proba'):",
    "        probs = model.predict_proba(X_test)",
    "        try:",
    "            auc = roc_auc_score(y_test, probs, multi_class='ovr')",
    "            print(f'ROC AUC (OvR): {auc:.4f}')",
    "        except:",
    "            print('ROC AUC: Error calculating')",
    "    print('\\n')"
])
add_code([
    "print('--- Clustering Silhouette Scores ---')",
    "for name, labels in clustering_results.items():",
    "    unique_labels = len(set(labels))",
    "    if unique_labels > 1 and unique_labels < len(X_transformed):",
    "        if name == 'SOM':",
    "             pass",
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
