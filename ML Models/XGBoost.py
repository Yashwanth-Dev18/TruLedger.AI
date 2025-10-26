import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


df = pd.read_csv("c:/Users/hp/LNU/TruLedger-AI/Datasets/Processed/TrainingSet.csv")
print(f"Dataset shape: {df.shape}")
print(f"Fraud rate: {df['is_fraud'].mean():.4f}")


X = df.drop('is_fraud', axis=1)
y = df['is_fraud']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    stratify=y,  # Important for imbalanced data!
    random_state=42
)

print(f"Train fraud rate: {y_train.mean():.4f}")
print(f"Test fraud rate: {y_test.mean():.4f}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

fraud_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
print(f"Fraud ratio (normal:fraud): {fraud_ratio:.1f}:1")

# Training XGBoost
print("\nTraining XGBoost...")
xgb_model = XGBClassifier(
    scale_pos_weight=fraud_ratio,  # Critical for imbalance!
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=20
)

xgb_model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    verbose=10
)


y_pred = xgb_model.predict(X_test_scaled)
y_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]  # Fraud probabilities

# Performance Evaluation
print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))
print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

# Feature Importance
feature_importance = xgb_model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False).head(20)

plt.figure(figsize=(10, 8))
sns.barplot(data=importance_df, y='feature', x='importance')
plt.title('Top 20 Features - XGBoost Importance')
plt.tight_layout()
plt.show()

print("\nTop 10 Most Important Features:")
print(importance_df.head(10))

# Precision-Recall Curve (Better for Imbalanced Data)
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

plt.figure(figsize=(10, 6))
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (AP = {avg_precision:.4f})')
plt.grid(True)
plt.show()

# Testing Different Thresholds
print("\n Testing Different Thresholds:")
for threshold in [0.3, 0.5, 0.7]:
    y_pred_custom = (y_pred_proba > threshold).astype(int)
    fraud_recall = recall_score(y_test, y_pred_custom, pos_label=1)
    print(f"Threshold {threshold}: Fraud Recall = {fraud_recall:.4f}")


joblib.dump(xgb_model, 'xgboost_fraud_model.pkl')
joblib.dump(scaler, 'xgboost_scaler.pkl')
print("\n💾 Models saved: 'xgboost_fraud_model.pkl' and 'xgboost_scaler.pkl'")

print("\n XGBoost Training Complete!")