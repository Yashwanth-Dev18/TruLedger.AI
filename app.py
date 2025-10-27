import streamlit as st
import pandas as pd
import numpy as np
import json
import os

from dataProcessor import process_transaction_data
from ML_Engine import run_fraud_detection

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'selected_dataset' not in st.session_state:
    st.session_state.selected_dataset = None
if 'fraud_df' not in st.session_state:
    st.session_state.fraud_df = pd.DataFrame()
if 'viz_data' not in st.session_state:
    st.session_state.viz_data = {}
if 'llm_explanations' not in st.session_state:
    st.session_state.llm_explanations = []
if 'current_explanation_page' not in st.session_state:
    st.session_state.current_explanation_page = 0

# =============================================
# PAGE CONFIGURATION
# =============================================

st.set_page_config(
    page_title="TruLedger - AI Fraud Detection",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================
# ENHANCED CSS STYLING
# =============================================

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styles */
    .main-header {
        font-size: 4rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0;
        background: linear-gradient(135deg, #0084ff 0%, #00d4ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        padding: 2rem 0 0.5rem 0;
        animation: fadeIn 1s ease-in;
    }
    
    .subtitle {
        font-size: 1.4rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
        animation: fadeIn 1.5s ease-in;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        margin: 3rem 0 1.5rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid #0084ff;
        animation: slideInLeft 0.5s ease-out;
    }
    
    /* Card Styles */
    .card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        margin: 1.5rem 0;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 132, 255, 0.15);
    }
    
    /* Tech Badge Styles */
    .tech-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 50px;
        margin: 0.5rem;
        font-weight: 600;
        font-size: 0.9rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .tech-badge:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Dataset Card */
    .dataset-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .dataset-card h3 {
        color: white;
        font-size: 1.8rem;
        margin-bottom: 1rem;
    }
    
    /* Metric Card */
    .metric-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #0084ff;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-left-color: #00d4ff;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0084ff;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #64748b;
        font-weight: 500;
    }
    
    /* Processing Step */
    .processing-step {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
        animation: pulse 2s ease-in-out infinite;
    }
    
    /* Fraud Alert Card */
    .fraud-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 20px rgba(255, 107, 107, 0.3);
        border-left: 5px solid #ff4757;
    }
    
    .fraud-card h4 {
        color: white;
        margin-bottom: 0.5rem;
    }
    
    /* LLM Explanation Card */
    .llm-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
        border-left: 5px solid #0084ff;
        transition: all 0.3s ease;
    }
    
    .llm-card:hover {
        box-shadow: 0 12px 30px rgba(0, 132, 255, 0.2);
        transform: translateX(5px);
    }
    
    /* Stats Highlight */
    .stats-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .stats-highlight h3 {
        color: white;
        font-size: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.02);
        }
    }
    
    /* Button Overrides */
    .stButton > button {
        background: linear-gradient(135deg, #0084ff 0%, #00d4ff 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 132, 255, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 132, 255, 0.4);
    }
    
    /* Selectbox Styling */
    .stSelectbox > div > div {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        font-size: 1.1rem;
    }
    
    /* Dataframe Styling */
    .dataframe {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

# =============================================
# DATA LOADING FUNCTIONS
# =============================================

@st.cache_data
def load_fraud_data():
    try:
        if os.path.exists('detected_fraud_transactions.csv'):
            return pd.read_csv('detected_fraud_transactions.csv')
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading fraud data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_visualization_data():
    try:
        if os.path.exists('app_visualization_data.json'):
            with open('app_visualization_data.json', 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        st.error(f"Error loading visualization data: {e}")
        return None

@st.cache_data
def load_llm_explanations():
    try:
        if os.path.exists('llm_fraud_explanations.json'):
            with open('llm_fraud_explanations.json', 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        st.error(f"Error loading LLM explanations: {e}")
        return None

# =============================================
# MAIN APP
# =============================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🏦 TruLedger</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Financial Reconciliation & Fraud Detection Assistant</p>', unsafe_allow_html=True)
    
    # Technology Stack
    st.markdown('<h2 class="section-header">🛠️ Technology Stack</h2>', unsafe_allow_html=True)
    
    tech_html = """
    <div style="text-align: center; margin: 2rem 0;">
        <span class="tech-badge">🔥 TensorFlow</span>
        <span class="tech-badge">📊 Scikit-learn</span>
        <span class="tech-badge">🤖 XGBoost</span>
        <span class="tech-badge">🧠 LangChain</span>
        <span class="tech-badge">💬 OpenAI</span>
        <span class="tech-badge">🐘 PostgreSQL</span>
        <span class="tech-badge">🐳 Docker</span>
        <span class="tech-badge">⚡ PySpark</span>
    </div>
    """
    st.markdown(tech_html, unsafe_allow_html=True)
    
    # Dataset Selection
    st.markdown('<h2 class="section-header">📁 Dataset Selection</h2>', unsafe_allow_html=True)
    
    dataset_options = {
        "TransactionLogs-1": {"desc": "Small business transactions", "records": "1,000 records"},
        "TransactionLogs-2": {"desc": "Medium enterprise transactions", "records": "5,000 records"},
        "TransactionLogs-3": {"desc": "Large financial logs", "records": "10,000 records"}
    }
    
    selected_key = st.selectbox(
        "🎯 Select Dataset",
        options=list(dataset_options.keys()),
        help="Choose a dataset to analyze"
    )
    
    if selected_key:
        info = dataset_options[selected_key]
        st.markdown(f"""
        <div class="dataset-card">
            <h3>📊 {selected_key}</h3>
            <p><strong>Description:</strong> {info['desc']}</p>
            <p><strong>Records:</strong> {info['records']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Run Pipeline Button
    if st.button("🚀 Run Complete Fraud Detection Pipeline", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        input_file = os.path.join("Uploaded_Datasets", "Raw", f"{selected_key}.csv")
        
        if not os.path.exists(input_file):
            st.error(f"❌ Dataset file not found: {input_file}")
            return
        
        # Step 1
        status_text.markdown('<div class="processing-step">🔄 Processing raw data...</div>', unsafe_allow_html=True)
        progress_bar.progress(25)
        processed_file = process_transaction_data(input_file)
        
        if processed_file is None:
            st.error("❌ Failed to process data")
            return
        
        # Step 2
        status_text.markdown('<div class="processing-step">🔍 Running fraud detection...</div>', unsafe_allow_html=True)
        progress_bar.progress(50)
        success = run_fraud_detection(processed_file)
        
        if not success:
            st.error("❌ Failed to run fraud detection")
            return
        
        # Step 3
        status_text.markdown('<div class="processing-step">📊 Loading results...</div>', unsafe_allow_html=True)
        progress_bar.progress(75)
        
        fraud_df = load_fraud_data()
        viz_data = load_visualization_data()
        llm_explanations = load_llm_explanations()
        
        progress_bar.progress(100)
        status_text.markdown('<div class="processing-step">✅ Analysis complete!</div>', unsafe_allow_html=True)
        
        if not fraud_df.empty:
            st.success(f"✅ Found {len(fraud_df)} suspicious transactions")
            st.session_state.analysis_complete = True
            st.session_state.fraud_df = fraud_df
            st.session_state.viz_data = viz_data
            st.session_state.llm_explanations = llm_explanations.get('explanations', []) if llm_explanations else []
    
    # Results Section
    if st.session_state.get('analysis_complete', False):
        # Model Performance
        st.markdown('<h2 class="section-header">🤖 ML Model Performance</h2>', unsafe_allow_html=True)
        
        viz_data = st.session_state.get('viz_data', {})
        model_perf = viz_data.get('model_performance', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = [
            ("Accuracy", model_perf.get('accuracy', 0)),
            ("Precision", model_perf.get('precision', 0)),
            ("Recall", model_perf.get('recall', 0)),
            ("F1-Score", model_perf.get('f1_score', 0))
        ]
        
        for col, (label, value) in zip([col1, col2, col3, col4], metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Stats Highlight
        st.markdown("""
        <div class="stats-highlight">
            <h3>🎯 Exceptional Detection Performance</h3>
            <p>Our XGBoost model achieves industry-leading fraud detection with minimal false positives</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Fraud Transactions
        st.markdown('<h2 class="section-header">🚨 Detected Fraud Transactions</h2>', unsafe_allow_html=True)
        fraud_df = st.session_state.get('fraud_df', pd.DataFrame())
        
        if not fraud_df.empty:
            st.dataframe(fraud_df.head(10), use_container_width=True, height=400)
        
        # LLM Explanations
        if st.session_state.get('llm_explanations'):
            st.markdown('<h2 class="section-header">🧠 AI-Powered Fraud Explanations</h2>', unsafe_allow_html=True)
            
            explanations = st.session_state.get('llm_explanations', [])
            
            for i, explanation in enumerate(explanations[:5]):
                transaction_id = explanation.get('transaction_id', i + 1)
                risk_factors = explanation.get('risk_factors', [])
                confidence = explanation.get('confidence', 'medium')
                explanation_text = explanation.get('explanation', 'No explanation available')
                
                confidence_color = {'high': '#ff6b6b', 'medium': '#f39c12', 'low': '#27ae60'}.get(confidence, '#64748b')
                
                st.markdown(f"""
                <div class="llm-card">
                    <h4>🚨 Transaction #{transaction_id}</h4>
                    <p><strong>Confidence:</strong> <span style="color: {confidence_color}; font-weight: 700;">{confidence.upper()}</span></p>
                    <p><strong>Risk Factors:</strong> {', '.join(risk_factors)}</p>
                    <p><strong>AI Explanation:</strong> {explanation_text}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #64748b; padding: 2rem; font-size: 0.9rem;'>
        <p style='font-weight: 600; font-size: 1.1rem;'>Developed by Yashwanth Krishna Devanaboina</p>
        <p>Built with cutting-edge AI technologies for explainable fraud detection</p>
        <p style='margin-top: 1rem;'>© 2024 TruLedger AI | Enterprise-Grade Fraud Detection</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
