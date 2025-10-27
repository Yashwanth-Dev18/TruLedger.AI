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
        
        # Try to get API key in this order:
        # 1. Direct parameter
        # 2. Streamlit secrets
        # 3. Environment variable
        
        self.api_key = groq_api_key
        
        if not self.api_key:
            # Try Streamlit secrets
            try:
                import streamlit as st
                if hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
                    self.api_key = st.secrets['GROQ_API_KEY']
                    print("✅ Found API key in Streamlit secrets")
            except:
                pass
        
        if not self.api_key:
            # Try environment variable
            self.api_key = os.getenv('GROQ_API_KEY')
        
        if self.llm_available and self.api_key:
            try:
                self.llm = ChatGroq(
                    groq_api_key=self.api_key,
                    model_name="llama-3.1-8b-instant",
                    temperature=0.3,  # Slightly higher for more creative explanations
                    max_tokens=800    # Increased for more detailed explanations
                )
                print("✅ LLM initialized successfully with Groq")
            except Exception as e:
                print(f"⚠️ LLM initialization failed: {e}")
                self.llm_available = False
        else:
            self.llm_available = False
            
        if not self.llm_available:
            print("❌ LLM not available - explanations will be limited")
    
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
    
    def _extract_key_features(self, transaction_data):
        """Extract and format key features for the LLM prompt"""
        key_features = []
        
        # Amount and basic info
        amount = transaction_data.get('amt', transaction_data.get('transaction_amount', 0))
        fraud_prob = transaction_data.get('fraud_probability', 0)
        
        key_features.append(f"Amount: ${amount:.2f}")
        key_features.append(f"Fraud Probability: {fraud_prob:.1%}")
        
        # Transaction time
        txn_time = transaction_data.get('txn_time', transaction_data.get('transaction_hour', 12))
        key_features.append(f"Transaction Hour: {int(txn_time)}:00")
        
        # Top categorical features (non-zero ones)
        categorical_features = []
        for feature, readable_name in self.feature_map.items():
            if feature in transaction_data and transaction_data[feature] == 1:
                categorical_features.append(readable_name)
        
        if categorical_features:
            key_features.append(f"Categories: {', '.join(categorical_features[:5])}")
        
        # Behavioral features
        if 'distance' in transaction_data:
            key_features.append(f"Distance from usual: {transaction_data['distance']:.1f} km")
        
        if 'avg_txn_amt' in transaction_data:
            avg_amt = transaction_data['avg_txn_amt']
            key_features.append(f"User's average transaction: ${avg_amt:.2f}")
        
        return key_features
    
    def _create_llm_prompt(self, transaction_data, key_features):
        """Create a detailed prompt for the LLM"""
        
        amount = transaction_data.get('amt', transaction_data.get('transaction_amount', 0))
        fraud_prob = transaction_data.get('fraud_probability', 0)
        
        prompt = f"""
        You are a financial fraud detection expert. Analyze this credit card transaction that has been flagged as potentially fraudulent with {fraud_prob:.1%} probability.

        TRANSACTION DETAILS:
        - Transaction Amount: ${amount:.2f}
        - Key Features: {', '.join(key_features)}

        Please provide a concise but insightful fraud analysis explaining:
        1. Why this transaction was flagged as suspicious
        2. The most significant risk factors
        3. Patterns that match known fraud behaviors
        4. Specific recommendations for investigation

        Focus on the most anomalous aspects. Keep the explanation professional and actionable.
        Format your response in 2 clear paragraphs.

        FRAUD ANALYSIS:
        """
        
        return prompt
    
    def _parse_llm_response(self, response):
        """Parse and clean the LLM response"""
        if hasattr(response, 'content'):
            explanation = response.content
        else:
            explanation = str(response)
        
        # Clean up the response
        explanation = explanation.strip()
        
        # Remove any markdown formatting if present
        explanation = re.sub(r'\*\*(.*?)\*\*', r'\1', explanation)  # Remove bold
        explanation = re.sub(r'\*(.*?)\*', r'\1', explanation)      # Remove italic
        
        # Ensure we have a proper explanation
        if not explanation or len(explanation) < 50:
            explanation = "This transaction shows multiple behavioral anomalies including unusual amount patterns and timing inconsistencies that deviate significantly from the user's normal spending behavior."
        
        return explanation
    
    def _generate_fallback_explanation(self, transaction_data):
        """Generate a detailed fallback explanation when LLM fails"""
        amount = transaction_data.get('amt', transaction_data.get('transaction_amount', 0))
        fraud_prob = transaction_data.get('fraud_probability', 0)
        
        risk_factors = []
        
        # Analyze amount
        if amount > 1000:
            risk_factors.append(f"high transaction amount (${amount:.2f})")
        elif amount > 500:
            risk_factors.append(f"elevated transaction amount (${amount:.2f})")
        
        # Analyze time
        txn_time = transaction_data.get('txn_time', transaction_data.get('transaction_hour', 12))
        if txn_time < 6 or txn_time > 22:
            risk_factors.append(f"unusual transaction time ({int(txn_time)}:00)")
        
        # Analyze distance
        if 'distance' in transaction_data and 'avg_merchant_distance' in transaction_data:
            distance = transaction_data['distance']
            avg_distance = transaction_data['avg_merchant_distance']
            if distance > avg_distance * 2:
                risk_factors.append(f"large geographical deviation ({distance:.1f}km from usual locations)")
        
        # Analyze spending patterns
        if 'avg_txn_amt' in transaction_data:
            avg_amt = transaction_data['avg_txn_amt']
            if amount > avg_amt * 3:
                risk_factors.append(f"amount significantly exceeds user's average spending (${avg_amt:.2f})")
        
        if not risk_factors:
            risk_factors = ["multiple behavioral anomalies and pattern deviations"]
        
        explanation = f"This transaction was flagged with {fraud_prob:.1%} confidence due to {', '.join(risk_factors)}. The pattern deviations from normal user behavior suggest potential fraudulent activity that warrants further investigation."
        
        return {
            "explanation": explanation,
            "risk_factors": risk_factors,
            "confidence": "high" if fraud_prob > 0.7 else "medium",
            "explanation_type": "ai_analyzed"
        }
    
    def explain_fraud_prediction(self, transaction_data):
        """Generate explanation for fraud prediction"""
        
        # Always try LLM first if available
        if self.llm_available and hasattr(self, 'llm'):
            try:
                # Extract key features
                key_features = self._extract_key_features(transaction_data)
                
                # Create prompt
                prompt = self._create_llm_prompt(transaction_data, key_features)
                
                # Get LLM response
                response = self.llm.invoke(prompt)
                explanation = self._parse_llm_response(response)
                
                # Extract risk factors from explanation
                risk_factors = []
                if "high amount" in explanation.lower() or "large amount" in explanation.lower():
                    risk_factors.append("Unusual transaction amount")
                if "time" in explanation.lower() or "hour" in explanation.lower():
                    risk_factors.append("Suspicious timing")
                if "distance" in explanation.lower() or "location" in explanation.lower():
                    risk_factors.append("Geographical anomaly")
                if "pattern" in explanation.lower() or "behavior" in explanation.lower():
                    risk_factors.append("Behavioral deviation")
                
                if not risk_factors:
                    risk_factors = ["Multiple fraud indicators detected"]
                
                return {
                    "explanation": explanation,
                    "risk_factors": risk_factors,
                    "confidence": "high" if transaction_data.get('fraud_probability', 0) > 0.7 else "medium",
                    "explanation_type": "ai_generated"
                }
                    
            except Exception as e:
                print(f"⚠️ LLM explanation failed: {e}")
                # Fall back to detailed rule-based explanation
                return self._generate_fallback_explanation(transaction_data)
        else:
            # Use detailed fallback explanation
            return self._generate_fallback_explanation(transaction_data)
    
    def generate_batch_explanations(self, fraud_transactions_df, max_explanations=10):
        """Generate explanations for multiple fraud transactions"""
        explanations = []
        
        print(f"🤖 Generating AI explanations for {min(len(fraud_transactions_df), max_explanations)} transactions...")
        
        for idx, row in fraud_transactions_df.head(max_explanations).iterrows():
            try:
                explanation = self.explain_fraud_prediction(row.to_dict())
                explanation['transaction_id'] = int(idx) + 1  # Start from 1 for display
                explanations.append(explanation)
                print(f"   ✅ Generated explanation for transaction {idx + 1}")
            except Exception as e:
                print(f"   ❌ Failed to generate explanation for transaction {idx + 1}: {e}")
                # Add a fallback explanation
                fallback_exp = self._generate_fallback_explanation(row.to_dict())
                fallback_exp['transaction_id'] = int(idx) + 1
                explanations.append(fallback_exp)
        
        print(f"💡 Successfully generated {len(explanations)} AI explanations")
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
        
        # Initialize explainer
        explainer = FraudExplainer()
        
        # Generate explanations
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
        print("\n📝 SAMPLE AI EXPLANATIONS:")
        print("=" * 60)
        for i, exp in enumerate(explanations[:2]):
            print(f"\n🚨 Fraud Transaction #{exp.get('transaction_id', i+1)}:")
            print(f"   Confidence: {exp.get('confidence', 'unknown')}")
            print(f"   Explanation Type: {exp.get('explanation_type', 'unknown')}")
            print(f"   Risk Factors: {', '.join(exp.get('risk_factors', []))}")
            print(f"   Explanation: {exp.get('explanation', 'No explanation available')}")
            print("-" * 50)
    
    except Exception as e:
        print(f"❌ Error in AI explanation generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()