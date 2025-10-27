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
                    temperature=0.2,  # Lower temperature for more consistent explanations
                    max_tokens=600    # Balanced token count
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
    
    def _extract_transaction_context(self, transaction_data):
        """Extract detailed transaction context for the LLM"""
        context = {}
        
        # Basic transaction info
        context['amount'] = transaction_data.get('amt', transaction_data.get('transaction_amount', 0))
        context['fraud_probability'] = transaction_data.get('fraud_probability', 0)
        context['transaction_hour'] = transaction_data.get('txn_time', transaction_data.get('transaction_hour', 12))
        
        # Extract categorical features that are active (value = 1)
        active_categories = []
        for feature, readable_name in self.feature_map.items():
            if feature in transaction_data and transaction_data[feature] == 1:
                active_categories.append(readable_name)
        context['active_categories'] = active_categories
        
        # Behavioral features
        context['distance'] = transaction_data.get('distance', None)
        context['avg_merchant_distance'] = transaction_data.get('avg_merchant_distance', None)
        context['avg_txn_amt'] = transaction_data.get('avg_txn_amt', None)
        context['txn_count'] = transaction_data.get('txn_count', None)
        
        # Time-based features
        context['days_since_last_txn'] = transaction_data.get('days_since_last_txn', None)
        context['avg_txn_time'] = transaction_data.get('avg_txn_time', None)
        
        return context
    
    def _create_detailed_prompt(self, transaction_context):
        """Create a detailed, specific prompt for fraud analysis"""
        
        amount = transaction_context['amount']
        fraud_prob = transaction_context['fraud_probability']
        hour = transaction_context['transaction_hour']
        
        prompt_parts = [
            "You are a forensic financial analyst. Analyze this flagged transaction and provide SPECIFIC reasons for fraud suspicion.",
            "",
            "TRANSACTION DETAILS:",
            f"- Amount: ${amount:.2f}",
            f"- Time: {int(hour):02d}:00",
        ]
        
        # Add categorical information
        if transaction_context['active_categories']:
            prompt_parts.append(f"- Categories: {', '.join(transaction_context['active_categories'])}")
        
        # Add behavioral anomalies
        anomalies = []
        
        # Amount anomalies
        if transaction_context['avg_txn_amt']:
            avg_amt = transaction_context['avg_txn_amt']
            amount_ratio = amount / avg_amt if avg_amt > 0 else 0
            if amount_ratio > 3:
                anomalies.append(f"Amount (${amount:.2f}) is {amount_ratio:.1f}x higher than user's average (${avg_amt:.2f})")
            elif amount_ratio > 2:
                anomalies.append(f"Amount (${amount:.2f}) is significantly above user's average (${avg_amt:.2f})")
        
        # Distance anomalies
        if transaction_context['distance'] and transaction_context['avg_merchant_distance']:
            distance = transaction_context['distance']
            avg_distance = transaction_context['avg_merchant_distance']
            distance_ratio = distance / avg_distance if avg_distance > 0 else 0
            if distance_ratio > 3:
                anomalies.append(f"Distance ({distance:.1f}km) is {distance_ratio:.1f}x farther than usual locations ({avg_distance:.1f}km)")
            elif distance_ratio > 2:
                anomalies.append(f"Unusual location distance ({distance:.1f}km vs usual {avg_distance:.1f}km)")
        
        # Time anomalies
        if transaction_context['avg_txn_time']:
            time_diff = abs(hour - transaction_context['avg_txn_time'])
            if time_diff > 6:
                anomalies.append(f"Unusual transaction time ({int(hour):02d}:00 vs usual {int(transaction_context['avg_txn_time']):02d}:00)")
        
        # Late night/early morning transactions
        if hour < 6 or hour > 22:
            anomalies.append(f"Transaction occurred during unusual hours ({int(hour):02d}:00)")
        
        if anomalies:
            prompt_parts.append("")
            prompt_parts.append("DETECTED ANOMALIES:")
            for anomaly in anomalies:
                prompt_parts.append(f"- {anomaly}")
        
        prompt_parts.extend([
            "",
            "ANALYSIS REQUIREMENTS:",
            "1. Be SPECIFIC about which transaction features triggered the fraud flag",
            "2. Explain WHY these features are suspicious in financial fraud context", 
            "3. Connect the dots between different anomalies",
            "4. Focus on concrete transaction attributes, not generic statements",
            "5. Do NOT mention confidence scores or probability percentages",
            "",
            "FRAUD ANALYSIS:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_llm_response(self, response):
        """Parse and clean the LLM response"""
        if hasattr(response, 'content'):
            explanation = response.content
        else:
            explanation = str(response)
        
        # Clean up the response
        explanation = explanation.strip()
        
        # Remove any markdown formatting and confidence mentions
        explanation = re.sub(r'\*\*(.*?)\*\*', r'\1', explanation)  # Remove bold
        explanation = re.sub(r'\*(.*?)\*', r'\1', explanation)      # Remove italic
        explanation = re.sub(r'confidence.*?\d+%', '', explanation, flags=re.IGNORECASE)  # Remove confidence mentions
        explanation = re.sub(r'probability.*?\d+%', '', explanation, flags=re.IGNORECASE)  # Remove probability mentions
        
        # Ensure we have a proper explanation
        if not explanation or len(explanation) < 50:
            explanation = "This transaction exhibits multiple behavioral anomalies including unusual transaction patterns, amount deviations from normal spending behavior, and timing inconsistencies that are characteristic of fraudulent activity."
        
        return explanation
    
    def _generate_specific_fallback(self, transaction_context):
        """Generate specific fallback explanation based on actual transaction data"""
        amount = transaction_context['amount']
        hour = transaction_context['transaction_hour']
        
        specific_reasons = []
        
        # Amount-based reasons
        if transaction_context['avg_txn_amt']:
            avg_amt = transaction_context['avg_txn_amt']
            if amount > avg_amt * 3:
                specific_reasons.append(f"transaction amount (${amount:.2f}) is more than 3 times higher than the user's typical spending (${avg_amt:.2f})")
            elif amount > avg_amt * 2:
                specific_reasons.append(f"unusually high amount of ${amount:.2f} compared to normal transactions around ${avg_amt:.2f}")
        
        # Location-based reasons
        if transaction_context['distance'] and transaction_context['avg_merchant_distance']:
            distance = transaction_context['distance']
            avg_distance = transaction_context['avg_merchant_distance']
            if distance > avg_distance * 2:
                specific_reasons.append(f"geographical distance of {distance:.1f}km from usual merchant locations (typically {avg_distance:.1f}km)")
        
        # Time-based reasons
        if hour < 6 or hour > 22:
            specific_reasons.append(f"transaction timing at {int(hour):02d}:00 outside normal spending hours")
        
        if transaction_context['avg_txn_time']:
            time_diff = abs(hour - transaction_context['avg_txn_time'])
            if time_diff > 6:
                specific_reasons.append(f"unusual transaction time compared to user's typical pattern")
        
        # Category-based reasons
        if transaction_context['active_categories']:
            categories = transaction_context['active_categories']
            if any(cat in ['Travel', 'Entertainment', 'Online Shopping'] for cat in categories):
                specific_reasons.append(f"high-risk transaction categories: {', '.join(categories)}")
        
        if not specific_reasons:
            specific_reasons = ["multiple behavioral pattern deviations from established spending history"]
        
        explanation = f"This transaction was flagged due to {', '.join(specific_reasons)}. These patterns are inconsistent with the account holder's typical financial behavior and match known fraud indicators."
        
        # Extract risk factors for display
        risk_factors = []
        if amount > 500: risk_factors.append("High Amount")
        if hour < 6 or hour > 22: risk_factors.append("Unusual Timing")
        if transaction_context['distance'] and transaction_context['distance'] > 50: risk_factors.append("Geographic Anomaly")
        if not risk_factors: risk_factors = ["Behavioral Pattern Deviation"]
        
        return {
            "explanation": explanation,
            "risk_factors": risk_factors,
            "explanation_type": "pattern_analysis"
        }
    
    def explain_fraud_prediction(self, transaction_data):
        """Generate specific fraud explanation"""
        
        # Extract detailed context
        transaction_context = self._extract_transaction_context(transaction_data)
        
        # Always try LLM first if available
        if self.llm_available and hasattr(self, 'llm'):
            try:
                # Create detailed prompt
                prompt = self._create_detailed_prompt(transaction_context)
                
                # Get LLM response
                response = self.llm.invoke(prompt)
                explanation = self._parse_llm_response(response)
                
                # Extract meaningful risk factors
                risk_factors = []
                if transaction_context['amount'] > 1000:
                    risk_factors.append("High Value Transaction")
                if transaction_context['transaction_hour'] < 6 or transaction_context['transaction_hour'] > 22:
                    risk_factors.append("Atypical Timing")
                if transaction_context['distance'] and transaction_context['distance'] > transaction_context.get('avg_merchant_distance', 0) * 2:
                    risk_factors.append("Location Anomaly")
                if transaction_context['active_categories'] and any(cat in ['Travel', 'Online Shopping'] for cat in transaction_context['active_categories']):
                    risk_factors.append("High-Risk Category")
                
                if not risk_factors:
                    risk_factors = ["Multiple Fraud Indicators"]
                
                return {
                    "explanation": explanation,
                    "risk_factors": risk_factors,
                    "explanation_type": "ai_analysis"
                }
                    
            except Exception as e:
                print(f"⚠️ LLM explanation failed: {e}")
                return self._generate_specific_fallback(transaction_context)
        else:
            return self._generate_specific_fallback(transaction_context)
    
    def generate_batch_explanations(self, fraud_transactions_df):
        """Generate explanations for ALL fraud transactions"""
        explanations = []
        
        print(f"🤖 Generating AI explanations for {len(fraud_transactions_df)} fraud transactions...")
        
        for idx, row in fraud_transactions_df.iterrows():
            try:
                explanation = self.explain_fraud_prediction(row.to_dict())
                explanation['transaction_id'] = int(idx) + 1  # Start from 1 for display
                explanations.append(explanation)
                
                if (idx + 1) % 10 == 0:  # Progress update every 10 transactions
                    print(f"   ✅ Generated {idx + 1}/{len(fraud_transactions_df)} explanations")
                    
            except Exception as e:
                print(f"   ❌ Failed to generate explanation for transaction {idx + 1}: {e}")
                # Add a fallback explanation
                transaction_context = self._extract_transaction_context(row.to_dict())
                fallback_exp = self._generate_specific_fallback(transaction_context)
                fallback_exp['transaction_id'] = int(idx) + 1
                explanations.append(fallback_exp)
        
        print(f"💡 Successfully generated {len(explanations)} AI explanations")
        return explanations