import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load Dataset
df = pd.read_csv("c:/Users/hp/LNU/TruLedger-AI/Datasets/Processed/TrainingSet.csv")
print(f"Dataset shape: {df.shape}")
print(f"Fraud rate: {df['is_fraud'].mean():.4f}")

# Split Features and Target
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Isolation Forest
print("\n🏃 Training Isolation Forest...")
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.02,  # Expected fraud rate (adjust based on your data)
    random_state=42,
    max_samples=256,
    n_jobs=-1
)

# Fit on ALL data (unsupervised)
iso_forest.fit(X_scaled)

# Predict Anomalies
# Isolation Forest returns: 1 for normal, -1 for anomalies
anomaly_scores = iso_forest.decision_function(X_scaled)
predictions = iso_forest.predict(X_scaled)

# Convert to binary (0=normal, 1=fraud)
y_pred = (predictions == -1).astype(int)

# Evaluate Performance
print("\n📊 CONFUSION MATRIX:")
print(confusion_matrix(y, y_pred))
print("\n📈 CLASSIFICATION REPORT:")
print(classification_report(y, y_pred))

# Visualize Anomaly Scores
plt.figure(figsize=(12, 5))

# Plot 1: Distribution of anomaly scores
plt.subplot(1, 2, 1)
sns.histplot(anomaly_scores, bins=50, kde=True)
plt.axvline(np.percentile(anomaly_scores, 95), color='red', linestyle='--', label='95th %ile')
plt.xlabel('Anomaly Score (lower = more anomalous)')
plt.title('Anomaly Score Distribution')
plt.legend()

# Plot 2: Scores by actual fraud status
plt.subplot(1, 2, 2)
sns.boxplot(x=y, y=anomaly_scores)
plt.xticks([0, 1], ['Normal', 'Fraud'])
plt.xlabel('Actual Class')
plt.ylabel('Anomaly Score')
plt.title('Anomaly Scores by Actual Class')

plt.tight_layout()
plt.show()

# Feature Importance (Isolation Forest)
feature_importance = np.mean(iso_forest.estimators_[0].feature_importances_ for estimator in iso_forest.estimators_)
feature_names = X.columns

# Top 20 most important features
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False).head(20)

plt.figure(figsize=(10, 8))
sns.barplot(data=importance_df, y='feature', x='importance')
plt.title('Top 20 Features - Isolation Forest Importance')
plt.tight_layout()
plt.show()

print("\n🎯 Top 10 Most Important Features:")
print(importance_df.head(10))

# Saving the Model - Finalized
joblib.dump(iso_forest, 'isolation_forest_fraud_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\n💾 Models saved: 'isolation_forest_fraud_model.pkl' and 'scaler.pkl'")