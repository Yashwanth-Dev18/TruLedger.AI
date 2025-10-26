import pandas as pd
import numpy as np
import json
import os
import re

try:
    from langchain_groq import ChatGroq
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️ langchain-groq not available, LLM explanations disabled")

class FraudExplainer:
    def __init__(self, groq_api_key=None):
        """Initialize the LLM for fraud explanations"""
        self.llm_available = LLM_AVAILABLE
        self.feature_map = self._create_feature_mapping()
        
        if self.llm_available and groq_api_key:
            try:
                self.llm = ChatGroq(
                    groq_api_key=groq_api_key,
                    model_name="llama-3.1-8b-instant",
                    temperature=0.1,
                    max_tokens=500
                )
            except Exception as e:
                print(f"⚠️ LLM initialization failed: {e}")
                self.llm_available = False
        else:
            self.llm_available = False
            
        if not self.llm_available:
            print("🔄 Using rule-based explanations")
    
    def _create_feature_mapping(self):
        """Create human-readable names for encoded features"""
        return {
            # Transaction categories
            'TXNctg_gas_transport': 'Gas & Transportation',
            'TXNctg_grocery_pos': 'Grocery Purchase',
            'TXNctg_shopping_pos': 'Shopping Purchase', 
            'TXNctg_personal_care': 'Personal Care',
            'TXNctg_entertainment': 'Entertainment',
            'TXNctg_travel': 'Travel',
            'TXNctg_misc_net': 'Online Miscellaneous',
            'TXNctg_kids_pets': 'Kids/Pets',
            'TXNctg_grocery_net': 'Online Grocery',
            'TXNctg_food_dining': 'Food & Dining',
            'TXNctg_health_fitness': 'Health & Fitness',
            'TXNctg_home': 'Home Goods',
            'TXNctg_misc_pos': 'In-Store Miscellaneous',
            'TXNctg_shopping_net': 'Online Shopping',
            
            # Job categories
            'JOBctg_Business_&_Management': 'Business & Management',
            'JOBctg_Healthcare_&_Medical': 'Healthcare & Medical',
            'JOBctg_Engineering_&_Technology': 'Engineering & Technology',
            'JOBctg_Education_&_Teaching': 'Education & Teaching',
            'JOBctg_Finance_&_Accounting': 'Finance & Accounting',
            
            # Age decades
            'dob_20s': 'Born in 1920s',
            'dob_30s': 'Born in 1930s',
            'dob_40s': 'Born in 1940s',
            'dob_50s': 'Born in 1950s',
            'dob_60s': 'Born in 1960s',
            'dob_70s': 'Born in 1970s',
            'dob_80s': 'Born in 1980s',
            'dob_90s': 'Born in 1990s',
            'dob_00s': 'Born in 2000s'
        }
    
    def _get_rule_based_explanation(self, transaction_data):
        """Generate rule-based explanations when LLM is unavailable"""
        explanations = []
        
        # Amount-based rules
        amount = transaction_data.get('amt', transaction_data.get('transaction_amount', 0))
        if amount > 1000:
            explanations.append(f"High transaction amount (${amount:.2f})")
        elif amount > 500:
            explanations.append(f"Moderately high amount (${amount:.2f})")
        
        # Time anomaly
        txn_time = transaction_data.get('txn_time', transaction_data.get('transaction_hour', 12))
        if txn_time < 6 or txn_time > 22:
            explanations.append(f"Unusual transaction time ({int(txn_time)}:00)")
        
        # Distance anomaly
        if 'distance' in transaction_data and 'avg_merchant_distance' in transaction_data:
            distance = transaction_data['distance']
            avg_distance = transaction_data['avg_merchant_distance']
            if distance > avg_distance * 2:
                explanations.append(f"Large distance from usual locations ({distance:.1f}km)")
        
        # Behavioral anomalies
        if 'avg_txn_amt' in transaction_data:
            avg_amt = transaction_data['avg_txn_amt']
            if amount > avg_amt * 3:
                explanations.append(f"Amount significantly higher than user's average (${avg_amt:.2f})")
        
        if len(explanations) == 0:
            explanations = ["Multiple behavioral anomalies detected"]
        
        return {
            "risk_factors": explanations,
            "confidence": "medium",
            "recommendation": "Review transaction details and consider contacting the cardholder",
            "explanation_type": "rule_based"
        }
    
    def explain_fraud_prediction(self, transaction_data):
        """Generate explanation for fraud prediction"""
        
        # Use LLM if available
        if self.llm_available and hasattr(self, 'llm'):
            try:
                # Extract key features for the prompt
                key_features = []
                for feature, readable_name in self.feature_map.items():
                    if feature in transaction_data and transaction_data[feature] != 0:
                        key_features.append(f"{readable_name}: {transaction_data[feature]}")
                
                # Limit to top features to avoid token limits
                key_features = key_features[:10]
                
                prompt = f"""
                Explain why this credit card transaction was flagged as potentially fraudulent:
                
                Transaction Details:
                - Amount: ${transaction_data.get('amt', transaction_data.get('transaction_amount', 0)):.2f}
                - Time: {transaction_data.get('txn_time', transaction_data.get('transaction_hour', 12))}:00
                - Fraud Probability: {transaction_data.get('fraud_probability', 0):.1%}
                
                Key Features:
                {chr(10).join(key_features)}
                
                Provide a concise explanation focusing on the most significant risk factors.
                """
                
                response = self.llm.invoke(prompt)
                explanation = response.content if hasattr(response, 'content') else str(response)
                
                return {
                    "explanation": explanation.strip(),
                    "explanation_type": "llm_generated",
                    "risk_factors": ["AI-generated explanation"],
                    "confidence": "high" if transaction_data.get('fraud_probability', 0) > 0.7 else "medium"
                }
                    
            except Exception as e:
                print(f"⚠️ LLM explanation failed: {e}")
                return self._get_rule_based_explanation(transaction_data)
        else:
            # Use rule-based approach
            return self._get_rule_based_explanation(transaction_data)
    
    def generate_batch_explanations(self, fraud_transactions_df, max_explanations=5):
        """Generate explanations for multiple fraud transactions"""
        explanations = []
        
        for idx, row in fraud_transactions_df.head(max_explanations).iterrows():
            explanation = self.explain_fraud_prediction(row.to_dict())
            explanation['transaction_id'] = int(idx)
            explanations.append(explanation)
        
        return explanations

def main():
    """Main function for standalone testing"""
    
    # Check if required files exist
    if not os.path.exists('detected_fraud_transactions.csv'):
        print("❌ 'detected_fraud_transactions.csv' not found. Run fraud detection first.")
        return
    
    try:
        # Load fraud transactions
        fraud_df = pd.read_csv('detected_fraud_transactions.csv')
        print(f"📊 Loaded {len(fraud_df)} fraud transactions")
        
        if fraud_df.empty:
            print("❌ No fraud transactions to explain")
            return
        
        # Initialize explainer (without API key for testing)
        explainer = FraudExplainer()
        
        # Generate explanations (limit to first 5 for testing)
        explanations = explainer.generate_batch_explanations(fraud_df, max_explanations=5)
        
        # Save explanations
        output_data = {
            'generated_at': pd.Timestamp.now().isoformat(),
            'total_transactions': len(explanations),
            'explanations': explanations
        }
        
        with open('llm_fraud_explanations.json', 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"💾 Saved {len(explanations)} explanations to 'llm_fraud_explanations.json'")
        
        # Display sample explanations
        print("\n📝 SAMPLE EXPLANATIONS:")
        print("=" * 50)
        for i, exp in enumerate(explanations[:2]):
            print(f"\n🚨 Fraud #{i+1}:")
            print(f"   Confidence: {exp.get('confidence', 'unknown')}")
            print(f"   Explanation: {exp.get('explanation', 'No explanation available')}")
            print(f"   Type: {exp.get('explanation_type', 'unknown')}")
    
    except Exception as e:
        print(f"❌ Error in LLM explanation generation: {e}")

if __name__ == "__main__":
    main()