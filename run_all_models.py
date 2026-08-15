#!/usr/bin/env python3
"""Run all models and generate results + diagrams for the thesis."""

import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# Setup
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12
os.makedirs("images", exist_ok=True)

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv('transactions.csv')
print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

# Show sample
print("\nFirst 5 rows:")
print(df.head().to_string())

# ============================================================
# PREPROCESSING
# ============================================================
print("\n" + "=" * 60)
print("PREPROCESSING")
print("=" * 60)

df_processed = df.copy()

# Drop identifiers
drop_cols = ['transaction_id', 'user_id']
df_processed = df_processed.drop(columns=drop_cols)
print(f"Dropped ID columns: {drop_cols}")

# Extract datetime features
df_processed['transaction_time'] = pd.to_datetime(df_processed['transaction_time'])
df_processed['hour'] = df_processed['transaction_time'].dt.hour
df_processed['dayofweek'] = df_processed['transaction_time'].dt.dayofweek
df_processed['month'] = df_processed['transaction_time'].dt.month
df_processed = df_processed.drop(columns=['transaction_time'])
print("Extracted temporal features: hour, dayofweek, month")

# Label encoding
cat_cols = ['country', 'bin_country', 'channel', 'merchant_category']
encoder = LabelEncoder()
for col in cat_cols:
    df_processed[col] = encoder.fit_transform(df_processed[col].astype(str))
print(f"Encoded categorical columns: {cat_cols}")

# Missing values
missing = df_processed.isnull().sum().sum()
print(f"Total Missing Values: {missing}")

# Target distribution
target_counts = df_processed['is_fraud'].value_counts()
target_pct = df_processed['is_fraud'].value_counts(normalize=True) * 100
print(f"Target Class Distribution:")
print(f"  Non-Fraud (0): {target_counts[0]} ({target_pct[0]:.2f}%)")
print(f"  Fraud (1):     {target_counts[1]} ({target_pct[1]:.2f}%)")

# ============================================================
# SPLIT & SCALE
# ============================================================
print("\n" + "=" * 60)
print("DATA SPLITTING & SCALING")
print("=" * 60)

X = df_processed.drop(columns=['is_fraud'])
y = df_processed['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
num_cols = ['account_age_days', 'total_transactions_user', 'avg_amount_user', 'amount', 'shipping_distance_km']

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

print(f"Training set: {X_train.shape}")
print(f"Testing set:  {X_test.shape}")
print(f"Fraud in training: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
print(f"Fraud in testing:  {y_test.sum()} ({y_test.mean()*100:.2f}%)")

# ============================================================
# MODEL TRAINING
# ============================================================
print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

results = []

def evaluate_model(name, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = auc(fpr, tpr)
    cm = confusion_matrix(y_true, y_pred)
    
    result = {
        'Model': name, 'Accuracy': acc, 'Precision': prec, 
        'Recall': rec, 'F1-Score': f1, 'AUC': auc_score,
        'TP': cm[1,1], 'TN': cm[0,0], 'FP': cm[0,1], 'FN': cm[1,0],
        'FPR': fpr, 'TPR': tpr
    }
    results.append(result)
    
    print(f"\n{name}:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc_score:.4f}")
    print(f"  Confusion Matrix: TP={cm[1,1]}, TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}")
    
    return result

# 1. Decision Tree (Ali Raza)
print("\nTraining Decision Tree...")
dt = DecisionTreeClassifier(max_depth=10, random_state=42)
dt.fit(X_train_scaled, y_train)
evaluate_model('Decision Tree', y_test, dt.predict(X_test_scaled), dt.predict_proba(X_test_scaled)[:, 1])

# 2. Linear Regression (Ali Raza)
print("\nTraining Linear Regression...")
lr_reg = LinearRegression()
lr_reg.fit(X_train_scaled, y_train)
y_prob_lin = lr_reg.predict(X_test_scaled)
y_prob_lin_clipped = np.clip(y_prob_lin, 0, 1)
y_pred_lin = (y_prob_lin_clipped >= 0.5).astype(int)
evaluate_model('Linear Regression', y_test, y_pred_lin, y_prob_lin_clipped)

# 3. XGBoost (Ali Raza)
print("\nTraining XGBoost...")
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train_scaled, y_train)
evaluate_model('XGBoost', y_test, xgb_model.predict(X_test_scaled), xgb_model.predict_proba(X_test_scaled)[:, 1])

# 4. Naive Bayes (Adil)
print("\nTraining Naive Bayes...")
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
evaluate_model('Naive Bayes', y_test, nb.predict(X_test_scaled), nb.predict_proba(X_test_scaled)[:, 1])

# 5. Random Forest (Adil / Sarif)
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
evaluate_model('Random Forest', y_test, rf.predict(X_test_scaled), rf.predict_proba(X_test_scaled)[:, 1])

# 6. Logistic Regression (Hafiz)
print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
evaluate_model('Logistic Regression', y_test, lr.predict(X_test_scaled), lr.predict_proba(X_test_scaled)[:, 1])

# 7. KNN (Hafiz)
print("\nTraining KNN...")
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(X_train_scaled, y_train)
evaluate_model('KNN', y_test, knn.predict(X_test_scaled), knn.predict_proba(X_test_scaled)[:, 1])

# 8. SVM (Hafiz) - trained on subset
print("\nTraining SVM (on 30,000 sample subset)...")
X_train_svm, _, y_train_svm, _ = train_test_split(
    X_train_scaled, y_train, train_size=30000, random_state=42, stratify=y_train
)
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train_svm, y_train_svm)
evaluate_model('SVM', y_test, svm.predict(X_test_scaled), svm.predict_proba(X_test_scaled)[:, 1])

# ============================================================
# RESULTS TABLE
# ============================================================
print("\n" + "=" * 60)
print("COMPREHENSIVE RESULTS TABLE")
print("=" * 60)

results_df = pd.DataFrame([{
    'Model': r['Model'], 'Accuracy': r['Accuracy'], 'Precision': r['Precision'],
    'Recall': r['Recall'], 'F1-Score': r['F1-Score'], 'AUC': r['AUC']
} for r in results]).sort_values('F1-Score', ascending=False).reset_index(drop=True)
results_df.index += 1

print(results_df.to_string())

# Save results
results_df.to_csv('images/model_results.csv', index=False)
print("\nResults saved to images/model_results.csv")

# ============================================================
# GENERATE DIAGRAMS
# ============================================================
print("\n" + "=" * 60)
print("GENERATING DIAGRAMS")
print("=" * 60)

# --- Figure 1: Class Distribution ---
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#2ecc71', '#e74c3c']
bars = ax.bar(['Non-Fraud (0)', 'Fraud (1)'], [target_counts[0], target_counts[1]], color=colors, edgecolor='black')
for bar, count, pct in zip(bars, [target_counts[0], target_counts[1]], [target_pct[0], target_pct[1]]):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1000, 
            f'{count:,}\n({pct:.2f}%)', ha='center', va='bottom', fontweight='bold')
ax.set_ylabel('Number of Transactions')
ax.set_title('Dataset Class Distribution (N=299,695)')
ax.set_ylim(0, max(target_counts) * 1.15)
plt.tight_layout()
plt.savefig('images/class_distribution.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: images/class_distribution.png")

# --- Figure 2: Feature Distributions ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for idx, col in enumerate(num_cols):
    ax = axes[idx // 3, idx % 3]
    ax.hist(df[col], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title(col, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
# Remove empty subplot
axes[1, 2].set_visible(False)
plt.suptitle('Distribution of Numerical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('images/feature_distributions.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: images/feature_distributions.png")

# --- Figure 3: Correlation Heatmap ---
fig, ax = plt.subplots(figsize=(12, 10))
corr = df_processed.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, annot_kws={'size': 8})
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('images/correlation_heatmap.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: images/correlation_heatmap.png")

# --- Figure 4: All Models Confusion Matrices ---
model_preds = {
    'Decision Tree': dt.predict(X_test_scaled),
    'Linear Regression': y_pred_lin,
    'XGBoost': xgb_model.predict(X_test_scaled),
    'Naive Bayes': nb.predict(X_test_scaled),
    'Random Forest': rf.predict(X_test_scaled),
    'Logistic Regression': lr.predict(X_test_scaled),
    'KNN': knn.predict(X_test_scaled),
    'SVM': svm.predict(X_test_scaled),
}

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
cmaps = ['Blues', 'Oranges', 'Greens', 'Reds', 'Purples', 'Blues', 'Oranges', 'Greens']
for idx, (name, preds) in enumerate(model_preds.items()):
    ax = axes[idx // 4, idx % 4]
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmaps[idx], ax=ax,
                xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'],
                annot_kws={'size': 12})
    ax.set_title(name, fontweight='bold', fontsize=11)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.suptitle('Confusion Matrices for All Eight Models', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('images/confusion_matrices_all.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: images/confusion_matrices_all.png")

# --- Figure 5: ROC Curves All Models ---
fig, ax = plt.subplots(figsize=(10, 8))
colors_roc = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22', '#34495e']
for idx, r in enumerate(results):
    ax.plot(r['FPR'], r['TPR'], label=f"{r['Model']} (AUC={r['AUC']:.4f})", 
            color=colors_roc[idx], linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve Comparison Across All Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/roc_curves_all.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: images/roc_curves_all.png")

# --- Figure 6: Performance Metrics Bar Chart ---
fig, axes = plt.subplots(1, 4, figsize=(20, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors_bar = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22', '#34495e']
for idx, metric in enumerate(metrics):
    ax = axes[idx]
    vals = results_df[metric].values
    names = results_df['Model'].values
    bars = ax.barh(range(len(names)), vals, color=colors_bar[:len(names)], edgecolor='black')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(metric, fontsize=11)
    ax.set_title(metric, fontweight='bold', fontsize=12)
    ax.set_xlim(0, 1.05)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2., f'{val:.3f}', 
                va='center', fontsize=8)
plt.suptitle('Performance Metrics Comparison Across All Models', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('images/metrics_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: images/metrics_comparison.png")

# --- Figure 7: AUC Comparison ---
fig, ax = plt.subplots(figsize=(10, 6))
aucs = [r['AUC'] for r in results]
names = [r['Model'] for r in results]
sorted_idx = np.argsort(aucs)[::-1]
sorted_names = [names[i] for i in sorted_idx]
sorted_aucs = [aucs[i] for i in sorted_idx]
bars = ax.bar(range(len(sorted_names)), sorted_aucs, color=colors_bar[:len(sorted_names)], edgecolor='black')
ax.set_xticks(range(len(sorted_names)))
ax.set_xticklabels(sorted_names, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('AUC-ROC Score')
ax.set_title('AUC-ROC Score Comparison', fontsize=14, fontweight='bold')
ax.set_ylim(0.5, 1.02)
for bar, val in zip(bars, sorted_aucs):
    ax.text(bar.get_x() + bar.get_width()/2., val + 0.005, f'{val:.4f}', 
            ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('images/auc_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: images/auc_comparison.png")

# --- Figure 8: Feature Importance (XGBoost) ---
fig, ax = plt.subplots(figsize=(10, 6))
imp = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=True)
imp.plot(kind='barh', color='steelblue', edgecolor='black', ax=ax)
ax.set_xlabel('Importance Score')
ax.set_title('Feature Importance (XGBoost)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('images/feature_importance_xgboost.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: images/feature_importance_xgboost.png")

# --- Figure 9: Feature Importance (Random Forest) ---
fig, ax = plt.subplots(figsize=(10, 6))
imp_rf = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
imp_rf.plot(kind='barh', color='forestgreen', edgecolor='black', ax=ax)
ax.set_xlabel('Importance Score')
ax.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('images/feature_importance_rf.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: images/feature_importance_rf.png")

# --- Figure 10: Precision-Recall Tradeoff ---
fig, ax = plt.subplots(figsize=(10, 6))
model_names = [r['Model'] for r in results]
precisions = [r['Precision'] for r in results]
recalls = [r['Recall'] for r in results]
f1s = [r['F1-Score'] for r in results]

x_pos = np.arange(len(model_names))
width = 0.25
ax.bar(x_pos - width, precisions, width, label='Precision', color='#3498db', edgecolor='black')
ax.bar(x_pos, recalls, width, label='Recall', color='#e74c3c', edgecolor='black')
ax.bar(x_pos + width, f1s, width, label='F1-Score', color='#2ecc71', edgecolor='black')
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Score')
ax.set_title('Precision, Recall, and F1-Score Comparison', fontsize=14, fontweight='bold')
ax.legend()
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig('images/precision_recall_f1.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: images/precision_recall_f1.png")

# --- Figure 11: Fraud by Hour ---
fig, ax = plt.subplots(figsize=(10, 5))
fraud_by_hour = df_processed[df_processed['is_fraud'] == 1]['hour'].value_counts().sort_index()
total_by_hour = df_processed['hour'].value_counts().sort_index()
fraud_rate_by_hour = (fraud_by_hour / total_by_hour * 100).fillna(0)
ax.bar(fraud_rate_by_hour.index, fraud_rate_by_hour.values, color='#e74c3c', edgecolor='black', alpha=0.8)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Fraud Rate (%)')
ax.set_title('Fraud Rate by Hour of Day', fontsize=14, fontweight='bold')
ax.set_xticks(range(0, 24))
plt.tight_layout()
plt.savefig('images/fraud_by_hour.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: images/fraud_by_hour.png")

# --- Figure 12: Fraud by Channel ---
fig, ax = plt.subplots(figsize=(8, 5))
channel_fraud = df.groupby('channel')['is_fraud'].agg(['sum', 'count'])
channel_fraud['rate'] = channel_fraud['sum'] / channel_fraud['count'] * 100
ax.bar(channel_fraud.index, channel_fraud['rate'], color=['#3498db', '#e74c3c', '#2ecc71'], edgecolor='black')
ax.set_xlabel('Channel')
ax.set_ylabel('Fraud Rate (%)')
ax.set_title('Fraud Rate by Transaction Channel', fontsize=14, fontweight='bold')
for i, (idx, row) in enumerate(channel_fraud.iterrows()):
    ax.text(i, row['rate'] + 0.02, f"{row['rate']:.2f}%", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('images/fraud_by_channel.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: images/fraud_by_channel.png")

# --- Figure 13: Implementation Pipeline Diagram ---
fig, ax = plt.subplots(figsize=(14, 4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 2)
ax.axis('off')

steps = [
    (0.5, 'Raw CSV\n(299,695 × 17)'),
    (2.0, 'Preprocessing\n(Encode, Scale)'),
    (3.5, 'Train/Test Split\n(80:20)'),
    (5.0, 'Model Training\n(8 Algorithms)'),
    (6.5, 'Evaluation\n(5 Metrics)'),
    (8.0, 'Visualisation\n(Charts & Tables)'),
]
colors_pipe = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c']

for i, (x, label) in enumerate(steps):
    rect = plt.Rectangle((x, 0.5), 1.3, 1, facecolor=colors_pipe[i], edgecolor='black', 
                          linewidth=2, alpha=0.8, zorder=2)
    ax.add_patch(rect)
    ax.text(x + 0.65, 1.0, label, ha='center', va='center', fontsize=8, fontweight='bold', zorder=3)
    if i < len(steps) - 1:
        ax.annotate('', xy=(steps[i+1][0], 1.0), xytext=(x + 1.3, 1.0),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))

ax.set_title('Implementation Pipeline: End-to-End Data Flow', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('images/implementation_pipeline.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: images/implementation_pipeline.png")

# --- Figure 14: Model Architecture Overview ---
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

# Title
ax.text(6, 7.5, 'Model Architecture Overview', ha='center', fontsize=16, fontweight='bold')

# Ensemble models (top row)
for i, (name, color) in enumerate([('XGBoost', '#2ecc71'), ('Random Forest', '#3498db'), ('Decision Tree', '#f39c12')]):
    rect = plt.Rectangle((1 + i*3.5, 5.5), 2.5, 1.2, facecolor=color, edgecolor='black', 
                          linewidth=2, alpha=0.8)
    ax.add_patch(rect)
    ax.text(1 + i*3.5 + 1.25, 6.1, name, ha='center', va='center', fontsize=11, fontweight='bold', color='white')

# Traditional models (bottom row)
for i, (name, color) in enumerate([
    ('Logistic\nRegression', '#9b59b6'), ('KNN', '#e74c3c'), 
    ('SVM', '#1abc9c'), ('Naive\nBayes', '#e67e22'), ('Linear\nRegression', '#34495e')
]):
    rect = plt.Rectangle((0.5 + i*2.2, 2.5), 1.8, 1.2, facecolor=color, edgecolor='black', 
                          linewidth=2, alpha=0.8)
    ax.add_patch(rect)
    ax.text(0.5 + i*2.2 + 0.9, 3.1, name, ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# Labels
ax.text(6, 5.0, 'Ensemble / Tree-Based Methods', ha='center', fontsize=10, style='italic')
ax.text(6, 2.0, 'Traditional / Linear / Distance-Based Methods', ha='center', fontsize=10, style='italic')

# Arrow from data to models
ax.annotate('Preprocessed Data\n(16 features, 239,756 training samples)', 
            xy=(6, 4.5), xytext=(6, 1.2),
            arrowprops=dict(arrowstyle='->', color='black', lw=2),
            ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('images/model_architecture.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: images/model_architecture.png")

print("\n" + "=" * 60)
print("ALL DIAGRAMS GENERATED SUCCESSFULLY")
print("=" * 60)

# Save detailed results as JSON
detailed_results = []
for r in results:
    detailed_results.append({
        'Model': r['Model'], 'Accuracy': float(r['Accuracy']), 'Precision': float(r['Precision']),
        'Recall': float(r['Recall']), 'F1-Score': float(r['F1-Score']), 'AUC': float(r['AUC']),
        'TP': int(r['TP']), 'TN': int(r['TN']), 'FP': int(r['FP']), 'FN': int(r['FN'])
    })
with open('images/detailed_results.json', 'w') as f:
    json.dump(detailed_results, f, indent=2)
print("Saved: images/detailed_results.json")
