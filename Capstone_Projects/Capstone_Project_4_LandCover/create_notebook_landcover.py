import json
import os
import pandas as pd
from sklearn.datasets import fetch_covtype

notebook_filename = 'Land_Cover_Analysis.ipynb'
dataset_filename = 'land_cover.csv'

# Step 1: Fetch and Save Data
print("Fetching Forest Cover Type dataset (this may take a moment)...")
covtype = fetch_covtype()
print("Data fetched. Saving to CSV...")
df_full = pd.DataFrame(covtype.data, columns=covtype.feature_names)
df_full['Cover_Type'] = covtype.target
# optimize: save only a subset if full is too huge for user's disk? 
# Full is ~581k rows, approx 100MB csv. Acceptable.
df_full.to_csv(dataset_filename, index=False)
print(f"Dataset saved to {dataset_filename}")

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
add_md(["# Capstone Project 4: Land Cover Classification", "## Forest Cover Type Dataset Analysis"])

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
add_md(["## Module 1: Data Cleaning", "Load and sample data (for performance). Apply Feature Imputing."])
add_code([
    "# Load Dataset",
    f"df = pd.read_csv('{dataset_filename}')",
    "print('Original Data Shape:', df.shape)",
    "",
    "# Sample the data to 20,000 rows for manageable execution time on local machine",
    "df_sample = df.sample(n=20000, random_state=42)",
    "print('Sampled Data Shape:', df_sample.shape)",
    "df_sample.head()"
])
add_code([
    "# Introduce missing values randomly (5%)",
    "np.random.seed(42)",
    "df_missing = df_sample.copy()",
    "cols_to_miss = df_missing.columns[:-1] # Exclude target",
    "mask = np.random.rand(*df_missing[cols_to_miss].shape) < 0.05",
    "df_missing[cols_to_miss] = df_missing[cols_to_miss].mask(mask)",
    "print('\\nMissing values count (Artificially Introduced):')",
    "print(df_missing.isnull().sum().sum())"
])
add_code([
    "# Separate features and target",
    "X_missing = df_missing.drop('Cover_Type', axis=1)",
    "y = df_missing['Cover_Type']",
    "",
    "# 1. Simple Imputer (Mean)",
    "simple_imputer = SimpleImputer(strategy='mean')",
    "X_simple = pd.DataFrame(simple_imputer.fit_transform(X_missing), columns=X_missing.columns)",
    "",
    "# 2. KNN Imputer (Using small subsets if needed, effectively fits on X_missing which is 20k rows - OK)",
    "knn_imputer = KNNImputer(n_neighbors=3)",
    "X_knn = pd.DataFrame(knn_imputer.fit_transform(X_missing), columns=X_missing.columns)",
    "",
    "# 3. Iterative Imputer",
    "iter_imputer = IterativeImputer(max_iter=10, random_state=0)",
    "X_iter = pd.DataFrame(iter_imputer.fit_transform(X_missing), columns=X_missing.columns)",
    "",
    "print('\\nImputation completed. Using Simple Imputed data for speed and stability.')",
    "df_clean = X_simple.copy()",
    "df_clean['Cover_Type'] = y.values"
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
    "print(stats[['Skewness', 'Skew Analysis']].head(10))"
])

# Module 3: Data Transformations
add_md(["## Module 3: Data Transformations", "Apply Scalers and Transformations."])
add_code([
    "X = df_clean.drop('Cover_Type', axis=1)",
    "# Cover_Type is 1-7, verify.",
    "y_encoded = df_clean['Cover_Type']",
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
    "    scaler.fit_transform(X)",
    "print('Scalers applied successfully.')",
    "",
    "# Transformations (Yeo-Johnson) - Applying on a subset of continuous columns to avoid issues with binary columns",
    "# The dataset has binary columns (Wilderness_Area, Soil_Type). We should only transform continuous ones.",
    "continuous_cols = [c for c in X.columns if not c.startswith('Wilderness') and not c.startswith('Soil')]",
    "print(f'Transforming {len(continuous_cols)} continuous columns...')",
    "",
    "pt = PowerTransformer(method='yeo-johnson')",
    "X_transformed = X.copy()",
    "X_transformed[continuous_cols] = pt.fit_transform(X[continuous_cols])",
    "",
    "# Standardize the whole thing for modeling",
    "scaler = StandardScaler()",
    "X_final = pd.DataFrame(scaler.fit_transform(X_transformed), columns=X.columns)",
    "",
    "print('\\nSkewness after Yeo-Johnson (Continuous Cols):')",
    "print(X_final[continuous_cols].skew())"
])

# Module 4: Univariate Visualization
add_md(["## Module 4: Univariate Data Visualization", "Focusing on top features for brevity."])
add_code([
    "plt.figure(figsize=(10, 6))",
    "sns.countplot(x='Cover_Type', data=df_clean)",
    "plt.title('Distribution of Cover Type (Target)')",
    "plt.show()"
])
add_code([
    "# Visualizing only first 6 continuous features",
    "viz_cols = continuous_cols[:6]",
    "plt.figure(figsize=(15, 10))",
    "for i, feature in enumerate(viz_cols):",
    "    plt.subplot(2, 3, i+1)",
    "    sns.histplot(df_clean[feature], kde=True)",
    "    plt.title(f'Histogram of {feature}')",
    "plt.tight_layout()",
    "plt.show()"
])
add_code([
    "plt.figure(figsize=(15, 10))",
    "for i, feature in enumerate(viz_cols):",
    "    plt.subplot(2, 3, i+1)",
    "    sns.boxplot(x=df_clean[feature])",
    "    plt.title(f'Boxplot of {feature}')",
    "plt.tight_layout()",
    "plt.show()"
])

# Module 5: Balancing Dataset
add_md(["## Module 5: Balancing Dataset", "Demonstrating Undersampling and SMOTE."])
add_code([
    "def calculate_entropy(y):",
    "    elements, counts = np.unique(y, return_counts=True)",
    "    probabilities = counts / len(y)",
    "    entropy = -np.sum(probabilities * np.log2(probabilities))",
    "    return entropy",
    "",
    "print('Original Class Distribution:', np.bincount(y_encoded)[1:])",
    "entropy_orig = calculate_entropy(y_encoded)",
    "print(f'Entropy (Original): {entropy_orig:.4f}')",
    "",
    "# 2. Undersampling",
    "rus = RandomUnderSampler(random_state=42)",
    "X_res_under, y_res_under = rus.fit_resample(X_final, y_encoded)",
    "entropy_under = calculate_entropy(y_res_under)",
    "print(f'Entropy (Undersampled): {entropy_under:.4f}')",
    "print('Undersampled Counts:', np.bincount(y_res_under)[1:])",
    "",
    "# 3. SMOTE",
    "# Note: SMOTE on 20k rows with many features is reasonably fast",
    "smote = SMOTE(random_state=42)",
    "X_res_smote, y_res_smote = smote.fit_resample(X_final, y_encoded)",
    "entropy_smote = calculate_entropy(y_res_smote)",
    "print(f'Entropy (SMOTE): {entropy_smote:.4f}')"
])

# Module 6: Supervised Learning
add_md(["## Module 6: Supervised Learning", "ID3, Bayes, MLP, Linear Regression."])
add_code([
    "# Using the SMOTE balanced data for training",
    "# Reduce size for ID3/MLP execution speed if needed, but 20k base -> SMOTE is ok.",
    "X_train, X_test, y_train, y_test = train_test_split(X_res_smote, y_res_smote, test_size=0.2, random_state=42)",
    "",
    "models = {}",
    "",
    "# a) Decision Tree (ID3 like)",
    "dt = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=42)",
    "dt.fit(X_train, y_train)",
    "models['Decision Tree'] = dt",
    "print('Decision Tree Rules (snippet):')",
    "print(export_text(dt, feature_names=list(X.columns))[:500])",
    "",
    "# b) Naive Bayes",
    "nb = GaussianNB()",
    "nb.fit(X_train, y_train)",
    "models['Naive Bayes'] = nb",
    "",
    "# c) MLP Classifier",
    "mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=200, random_state=42)",
    "mlp.fit(X_train, y_train)",
    "models['MLP Classifier'] = mlp",
    "",
    "# d) Linear Regression",
    "lr = LinearRegression()",
    "lr.fit(X_train, y_train)",
    "models['Linear Regression'] = lr"
])

# Module 7: Unsupervised Learning
add_md(["## Module 7: Unsupervised Learning", "K-Means, SOM, DBSCAN, Mean Shift."])
add_code([
    "clustering_results = {}",
    "# Sampling further for expensive clustering algorithms (DBSCAN, MeanShift)",
    "# O(N^2) complexity makes them slow on >10k points.",
    "sample_cluster_mask = np.random.choice(X_final.index, 2000, replace=False)",
    "X_cluster = X_final.loc[sample_cluster_mask]",
    "",
    "# a) K-Means (7 clusters for 7 cover types)",
    "kmeans = KMeans(n_clusters=7, random_state=42)",
    "labels_kmeans = kmeans.fit_predict(X_cluster)",
    "clustering_results['KMeans'] = labels_kmeans",
    "",
    "# b) SOM",
    "som = MiniSom(x=10, y=10, input_len=X_cluster.shape[1], sigma=1.0, learning_rate=0.5)",
    "som.random_weights_init(X_cluster.values)",
    "som.train_random(X_cluster.values, 100)",
    "labels_som = [som.winner(x) for x in X_cluster.values]",
    "clustering_results['SOM'] = [str(x) for x in labels_som]",
    "",
    "# c) DBSCAN",
    "dbscan = DBSCAN(eps=3.0, min_samples=5)",
    "labels_dbscan = dbscan.fit_predict(X_cluster)",
    "clustering_results['DBSCAN'] = labels_dbscan",
    "",
    "# d) Mean Shift",
    "meanshift = MeanShift()",
    "labels_meanshift = meanshift.fit_predict(X_cluster)",
    "clustering_results['MeanShift'] = labels_meanshift"
])

# Module 8: Performance Metrics
add_md(["## Module 8: Performance Metrics", "Metrics for Supervised models."])
add_code([
    "for name, model in models.items():",
    "    print(f'--- {name} ---')",
    "    if name == 'Linear Regression':",
    "        y_pred = np.clip(np.rint(model.predict(X_test)), 1, 7).astype(int)",
    "    else:",
    "        y_pred = model.predict(X_test)",
    "    ",
    "    print('Accuracy:', accuracy_score(y_test, y_pred))",
    "    print('F1 Score (Weighted):', f1_score(y_test, y_pred, average='weighted'))",
    "    print('\\n')",
    "",
    "print('--- Clustering Silhouette Scores (on 2k subset) ---')",
    "for name, labels in clustering_results.items():",
    "    if len(set(labels)) > 1 and len(set(labels)) < len(X_cluster):",
    "         if name != 'SOM':",
    "             print(f'{name}: {silhouette_score(X_cluster, labels):.4f}')"
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
