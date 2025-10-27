import pandas as pd
import numpy as np
import os
import json
from LLM_XAI import FraudExplainer

try:
    import joblib
    from sklearn.preprocessing import StandardScaler
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("⚠️ joblib or scikit-learn not available, using fallback methods")

class SimpleFallbackDetector:
    """Fallback detector when model files are missing"""
    
    def predict_proba(self, X):
        """Predict fraud probabilities using simple rules"""
        probas = []
        
        # Convert to DataFrame if it's a numpy array
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X
            
        for _, row in X_df.iterrows():
            risk_score = 0.0
            
            # Amount-based risk
            amount = row.get('amt', 0) if hasattr(row, 'get') else (row[0] if len(row) > 0 else 0)
            if amount > 1000:
                risk_score += 0.6
            elif amount > 500:
                risk_score += 0.4
            elif amount > 200:
                risk_score += 0.2
            
            # Time anomaly risk
            txn_time_idx = 1 if len(row) > 1 else 0
            avg_txn_time_idx = 2 if len(row) > 2 else 0
            
            txn_time = row.get('txn_time', 12) if hasattr(row, 'get') else (row[txn_time_idx] if len(row) > txn_time_idx else 12)
            avg_txn_time = row.get('avg_txn_time', 12) if hasattr(row, 'get') else (row[avg_txn_time_idx] if len(row) > avg_txn_time_idx else 12)
            
            time_diff = abs(txn_time - avg_txn_time)
            if time_diff > 8:
                risk_score += 0.3
            elif time_diff > 4:
                risk_score += 0.15
            
            # Distance anomaly risk (if available)
            if hasattr(row, 'get'):
                if 'distance' in row and 'avg_merchant_distance' in row:
                    distance = row['distance']
                    avg_distance = row['avg_merchant_distance']
                    if distance > avg_distance * 3:
                        risk_score += 0.4
                    elif distance > avg_distance * 2:
                        risk_score += 0.2
            
            # Cap at 0.95
            risk_score = min(risk_score, 0.95)
            probas.append([1 - risk_score, risk_score])
        
        return np.array(probas)
    
    def predict(self, X):
        """Predict fraud labels"""
        probas = self.predict_proba(X)[:, 1]
        return (probas > 0.35).astype(int)

def load_components():
    """Load model components with fallbacks"""
    components = {}
    
    # Try to load pre-trained model
    if JOBLIB_AVAILABLE:
        model_files = {
            'model': 'xgboost_fraud_model.pkl',
            'scaler': 'xgboost_scaler.pkl',
            'threshold': 'optimal_threshold.pkl'
        }
        
        try:
            for key, filename in model_files.items():
                if os.path.exists(filename):
                    components[key] = joblib.load(filename)
                    print(f"✅ Loaded {filename}")
                else:
                    raise FileNotFoundError(f"{filename} not found")
            
            print("✅ Loaded pre-trained model components")
        except Exception as e:
            print(f"⚠️ Error loading model files: {e}")
            components['model'] = SimpleFallbackDetector()
            components['scaler'] = None
            components['threshold'] = 0.35
    else:
        print("🔄 Using fallback fraud detector")
        components['model'] = SimpleFallbackDetector()
        components['scaler'] = None
        components['threshold'] = 0.35
    
    return components

def run_fraud_detection(processed_file_path):
    """Run fraud detection on processed data"""
    print(f"🔍 Running fraud detection on: {processed_file_path}")
    
    try:
        # Load data
        df = pd.read_csv(processed_file_path)
        print(f"📊 Dataset shape: {df.shape}")
        
        # Prepare features
        if 'is_fraud' in df.columns:
            X = df.drop('is_fraud', axis=1)
            y_true = df['is_fraud']
        else:
            X = df.copy()
            y_true = pd.Series([0] * len(df))
        
        # Load model components
        components = load_components()
        
        # Scale features if scaler is available
        if components['scaler'] is not None and hasattr(components['scaler'], 'transform'):
            try:
                X_scaled = components['scaler'].transform(X)
            except Exception as e:
                print(f"⚠️ Scaling failed: {e}, using original features")
                X_scaled = X.values
        else:
            X_scaled = X.values
        
        # Get predictions - ensure we pass DataFrame to fallback detector
        if isinstance(components['model'], SimpleFallbackDetector):
            # For fallback detector, use DataFrame
            y_proba = components['model'].predict_proba(X)
        else:
            # For real model, use scaled features
            y_proba = components['model'].predict_proba(X_scaled)
        
        # Handle different probability array shapes
        if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
            fraud_proba = y_proba[:, 1]
        else:
            fraud_proba = y_proba.flatten() if y_proba.ndim > 1 else y_proba
        
        y_pred = (fraud_proba > components['threshold']).astype(int)
        
        print(f"🎯 Fraud detection results:")
        print(f"   - Flagged {y_pred.sum()} transactions as fraudulent")
        print(f"   - Detection rate: {y_pred.sum()/len(y_pred)*100:.2f}%")
        
        # Create fraud transactions dataset
        fraud_mask = y_pred == 1
        if fraud_mask.any():
            fraud_transactions = X[fraud_mask].copy()
            fraud_transactions['fraud_probability'] = fraud_proba[fraud_mask]
            fraud_transactions['is_fraud_predicted'] = 1
            
            # Ensure required columns for display
            if 'amt' in fraud_transactions.columns:
                fraud_transactions['transaction_amount'] = fraud_transactions['amt']
            if 'txn_time' in fraud_transactions.columns:
                fraud_transactions['transaction_hour'] = fraud_transactions['txn_time']
            
            fraud_transactions.to_csv('detected_fraud_transactions.csv', index=False)
            print(f"💾 Saved {len(fraud_transactions)} fraud transactions")
        else:
            # Create empty DataFrame with expected structure
            empty_fraud = pd.DataFrame(columns=['transaction_amount', 'transaction_hour', 'fraud_probability'])
            empty_fraud.to_csv('detected_fraud_transactions.csv', index=False)
            print("💾 No fraud detected - created empty results file")
        
        # Generate visualization data
        viz_data = generate_visualization_data(df, fraud_mask, fraud_proba)
        with open('app_visualization_data.json', 'w') as f:
            json.dump(viz_data, f, indent=2)
    
        print("🤖 Generating fraud explanations...")
        explanation_success = generate_fraud_explanations()
        
        if explanation_success:
            print("✅ Fraud explanations generated successfully")
        else:
            print("⚠️ Using rule-based explanations only")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in fraud detection: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_visualization_data(df, fraud_mask, fraud_proba=None):
    """Generate visualization data for the app"""
    
    if fraud_mask is None or fraud_mask.sum() == 0:
        # Sample data when no fraud detected
        return {
            'job_categories': {'Business': 1, 'Technology': 1},
            'transaction_categories': {'Shopping': 1, 'Food': 1},
            'age_groups': {'80s': 1, '90s': 1},
            'amount_analysis': {
                'fraud_avg': 0,
                'normal_avg': float(df.get('amt', 100).mean() if 'amt' in df.columns else 100),
                'increase_pct': 0
            },
            'model_performance': {
                'fraud_detected': 0,
                'total_transactions': len(df),
                'detection_rate': 0.0
            }
        }
    
    # Real data when fraud is detected
    fraud_df = df[fraud_mask]
    normal_df = df[~fraud_mask]
    
    # Job categories
    job_cols = [col for col in df.columns if col.startswith('JOBctg_')]
    job_data = {col: int(fraud_df[col].sum()) for col in job_cols[:5]} if job_cols else {'Sample_Job': 1}
    
    # Transaction categories  
    txn_cols = [col for col in df.columns if col.startswith('TXNctg_')]
    txn_data = {col: int(fraud_df[col].sum()) for col in txn_cols[:5]} if txn_cols else {'Sample_Category': 1}
    
    # Age groups
    age_cols = [col for col in df.columns if col.startswith('dob_')]
    age_data = {col: int(fraud_df[col].sum()) for col in age_cols[:4]} if age_cols else {'80s': 1}
    
    # Amount analysis
    fraud_avg = float(fraud_df['amt'].mean()) if 'amt' in fraud_df.columns else 500.0
    normal_avg = float(normal_df['amt'].mean()) if 'amt' in normal_df.columns else 100.0
    increase_pct = ((fraud_avg - normal_avg) / normal_avg * 100) if normal_avg > 0 else 0
    
    return {
        'job_categories': job_data,
        'transaction_categories': txn_data,
        'age_groups': age_data,
        'amount_analysis': {
            'fraud_avg': fraud_avg,
            'normal_avg': normal_avg,
            'increase_pct': increase_pct
        },
        'model_performance': {
            'fraud_detected': int(fraud_mask.sum()),
            'total_transactions': len(df),
            'detection_rate': float(fraud_mask.sum() / len(df))
        }
    }


# Then add this function to ML_Engine.py
def generate_fraud_explanations(fraud_df_path='detected_fraud_transactions.csv'):
    """Generate LLM explanations for detected fraud transactions"""
    try:
        from LLM_XAI import FraudExplainer
        
        # Check if fraud transactions exist
        if not os.path.exists(fraud_df_path):
            print("❌ No fraud transactions found for explanations")
            return False
        
        fraud_df = pd.read_csv(fraud_df_path)
        
        if fraud_df.empty:
            print("❌ No fraud transactions to explain")
            return False
        
        # Initialize explainer (without API key - will use rule-based)
        explainer = FraudExplainer()
        
        # Generate explanations
        explanations = explainer.generate_batch_explanations(fraud_df, max_explanations=10)
        
        # Save explanations
        output_data = {
            'generated_at': pd.Timestamp.now().isoformat(),
            'total_transactions': len(explanations),
            'explanations': explanations
        }
        
        with open('llm_fraud_explanations.json', 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"💾 Generated {len(explanations)} fraud explanations")
        return True
        
    except Exception as e:
        print(f"❌ Error generating fraud explanations: {e}")
        return False



# Standalone testing
if __name__ == "__main__":
    # Test with sample processed data
    test_file = os.path.join("Uploaded_Datasets", "Processed", "Processed_TransactionLogs-1.csv")
    if os.path.exists(test_file):
        result = run_fraud_detection(test_file)
        if result:
            print("✅ Fraud detection test completed successfully!")
    else:
        print("ℹ️ No processed test file found. Run data processing first.")