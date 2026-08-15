import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')

# ==============================================================================
# STEP 1: IMPORT LIBRARIES & LOAD DATASET
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

print("--- 1. LOADING DATASET ---")
# Ensure 'transactions.csv' is uploaded to your Google Colab environment
file_path = 'transactions.csv'
df = pd.read_csv(file_path)

print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
# display(df.head())

# ==============================================================================
# STEP 2: DATA PREPROCESSING (IN DETAIL)
# ==============================================================================
print("\n--- 2. DATA PREPROCESSING ---")

df_processed = df.copy()

# A. Drop non-predictive identifiers
drop_cols = ['transaction_id', 'user_id']
df_processed = df_processed.drop(columns=drop_cols)
print(f"-> Dropped ID columns: {drop_cols}")

# B. Extract datetime features
df_processed['transaction_time'] = pd.to_datetime(df_processed['transaction_time'])
df_processed['hour'] = df_processed['transaction_time'].dt.hour
df_processed['dayofweek'] = df_processed['transaction_time'].dt.dayofweek
df_processed['month'] = df_processed['transaction_time'].dt.month
df_processed = df_processed.drop(columns=['transaction_time'])
print("-> Extracted temporal features: 'hour', 'dayofweek', 'month'")

# C. Categorical Encoding (Label Encoding)
cat_cols = ['country', 'bin_country', 'channel', 'merchant_category']
encoder = LabelEncoder()

for col in cat_cols:
    df_processed[col] = encoder.fit_transform(df_processed[col].astype(str))
print(f"-> Encoded categorical columns: {cat_cols}")

# D. Check Missing Values & Target Imbalance
missing = df_processed.isnull().sum().sum()
print(f"-> Total Missing Values: {missing}")

target_counts = df_processed['is_fraud'].value_counts(normalize=True) * 100
print(f"-> Target Class Distribution ('is_fraud'):\n   0 (Non-Fraud): {target_counts[0]:.2f}%\n   1 (Fraud): {target_counts[1]:.2f}%")

# ==============================================================================
# STEP 3: DATA SPLIT & FEATURE SCALING (CRITICAL FOR SVM/KNN/LR)
# ==============================================================================
print("\n--- 3. DATA SPLITTING & SCALING ---")

X = df_processed.drop(columns=['is_fraud'])
y = df_processed['is_fraud']

# Stratified Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling is mandatory for Logistic Regression, SVM, and KNN
scaler = StandardScaler()
num_cols = ['account_age_days', 'total_transactions_user', 'avg_amount_user', 'amount', 'shipping_distance_km']

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

print(f"Training set dimensions: {X_train.shape}")
print(f"Testing set dimensions:  {X_test.shape}")

# ==============================================================================
# STEP 4: MODEL IMPLEMENTATION
# ==============================================================================
print("\n--- 4. MODEL IMPLEMENTATION & TRAINING ---")

# 1. Logistic Regression
print("Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# 2. K-Nearest Neighbors (KNN)
print("Training K-Nearest Neighbors (KNN)...")
knn_model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_model.fit(X_train, y_train)

# 3. Support Vector Machine (SVM)
# Note: Training SVC on full large datasets can be resource-heavy.
# We train on an optimized stratified subset (e.g., 30,000 samples) to ensure fast execution in Colab.
print("Training Support Vector Machine (SVM)...")
X_train_svm, _, y_train_svm, _ = train_test_split(
    X_train, y_train, train_size=30000, random_state=42, stratify=y_train
)
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train_svm, y_train_svm)

# Make Predictions
y_pred_lr = lr_model.predict(X_test)
y_prob_lr = lr_model.predict_proba(X_test)[:, 1]

y_pred_knn = knn_model.predict(X_test)
y_prob_knn = knn_model.predict_proba(X_test)[:, 1]

y_pred_svm = svm_model.predict(X_test)
y_prob_svm = svm_model.predict_proba(X_test)[:, 1]

# ==============================================================================
# STEP 5: MODEL EVALUATION & RANK TABLE
# ==============================================================================
print("\n--- 5. MODEL EVALUATION & RANK TABLE ---")

def evaluate_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1}

results = [
    evaluate_model('Logistic Regression', y_test, y_pred_lr),
    evaluate_model('K-Nearest Neighbors', y_test, y_pred_knn),
    evaluate_model('Support Vector Machine', y_test, y_pred_svm)
]

# Create Rank Table sorted by F1-Score
rank_df = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False).reset_index(drop=True)
rank_df.index += 1
rank_df.index.name = 'Rank'

print("\n=================== MODEL RANK TABLE ===================")
# display(rank_df.style.format({'Accuracy': '{:.4f}', 'Precision': '{:.4f}', 'Recall': '{:.4f}', 'F1-Score': '{:.4f}'}))

# Detailed Classification Reports
print("\n--- Classification Report: Logistic Regression ---")
print(classification_report(y_test, y_pred_lr, target_names=['Non-Fraud', 'Fraud']))

print("\n--- Classification Report: K-Nearest Neighbors ---")
print(classification_report(y_test, y_pred_knn, target_names=['Non-Fraud', 'Fraud']))

print("\n--- Classification Report: Support Vector Machine ---")
print(classification_report(y_test, y_pred_svm, target_names=['Non-Fraud', 'Fraud']))

# ==============================================================================
# STEP 6: VISUALIZATIONS
# ==============================================================================
print("\n--- 6. GENERATING VISUALIZATIONS ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Confusion Matrix - Logistic Regression
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
axes[0, 0].set_title('Confusion Matrix: Logistic Regression')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')

# Plot 2: Confusion Matrix - KNN
sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', cmap='Purples', ax=axes[0, 1],
            xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
axes[0, 1].set_title('Confusion Matrix: KNN')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')

# Plot 3: Metrics Comparison Bar Chart
metrics_melted = rank_df.melt(id_vars=['Model'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                              var_name='Metric', value_name='Score')
sns.barplot(data=metrics_melted, x='Metric', y='Score', hue='Model', palette='Set2', ax=axes[1, 0])
axes[1, 0].set_title('Model Performance Metrics Comparison')
axes[1, 0].set_ylim(0.0, 1.05)
for p in axes[1, 0].patches:
    if p.get_height() > 0:
        axes[1, 0].annotate(f"{p.get_height():.3f}",
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 5),
                            textcoords='offset points', fontsize=7)

# Plot 4: ROC Curves
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
auc_lr = auc(fpr_lr, tpr_lr)

fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)
auc_knn = auc(fpr_knn, tpr_knn)

fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
auc_svm = auc(fpr_svm, tpr_svm)

axes[1, 1].plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.4f})', color='blue')
axes[1, 1].plot(fpr_knn, tpr_knn, label=f'KNN (AUC = {auc_knn:.4f})', color='purple')
axes[1, 1].plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {auc_svm:.4f})', color='green')
axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Random Chance')
axes[1, 1].set_title('ROC Curve Comparison')
axes[1, 1].set_xlabel('False Positive Rate')
axes[1, 1].set_ylabel('True Positive Rate')
axes[1, 1].legend(loc='lower right')

plt.tight_layout()
plt.show()



