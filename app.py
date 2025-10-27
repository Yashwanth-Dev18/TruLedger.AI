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
# DARK THEME CSS STYLING
# =============================================

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global Styles - BLACK BACKGROUND */
    .stApp {
        background: #000000;
        font-family: 'Inter', sans-serif;
        color: #ffffff;
    }
    
    /* Main content area background */
    .main .block-container {
        background: #000000;
        padding: 2rem;
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
        color: #94a3b8;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
        animation: fadeIn 1.5s ease-in;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 2rem;
        font-weight: 700;
        color: #00d4ff;
        margin: 3rem 0 1.5rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid #0084ff;
        animation: slideInLeft 0.5s ease-out;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
    }
    
    /* Dark Section Header */
    .dark-section-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00d4ff;
        margin: 2rem 0 1.5rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid #00d4ff;
        text-align: center;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
    }
    
    /* Card Styles - DARK THEME */
    .card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        margin: 1.5rem 0;
        border: 1px solid #2d3746;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 132, 255, 0.3);
        border-color: #0084ff;
    }
    
    /* Dark Theme Cards */
    .dark-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        border: 1px solid #2d3746;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
    }
    
    .dark-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 212, 255, 0.2);
        border-color: #00d4ff;
    }
    
    /* Dark Metric Card */
    .dark-metric-card {
        background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem;
        border: 1px solid #2d3746;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.6);
        transition: all 0.3s ease;
    }
    
    .dark-metric-card:hover {
        transform: translateY(-5px);
        border-color: #0084ff;
        box-shadow: 0 12px 35px rgba(0, 132, 255, 0.4);
    }
    
    .dark-metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00d4ff;
        margin: 0.5rem 0;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
    }
    
    .dark-metric-label {
        font-size: 1rem;
        color: #94a3b8;
        font-weight: 500;
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
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .tech-badge:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Dataset Card */
    .dataset-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .dataset-card h3 {
        color: white;
        font-size: 1.8rem;
        margin-bottom: 1rem;
    }
    
    /* Metric Card - DARK THEME */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.6);
        border-left: 4px solid #0084ff;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-left-color: #00d4ff;
        box-shadow: 0 12px 35px rgba(0, 132, 255, 0.4);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00d4ff;
        margin: 0.5rem 0;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
    }
    
    .metric-label {
        font-size: 1rem;
        color: #94a3b8;
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
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);
        animation: pulse 2s ease-in-out infinite;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Fraud Alert Card */
    .fraud-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 20px rgba(255, 107, 107, 0.4);
        border-left: 5px solid #ff4757;
    }
    
    .fraud-card h4 {
        color: white;
        margin-bottom: 0.5rem;
    }
    
    /* LLM Explanation Card - DARK THEME */
    .llm-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.6);
        border-left: 5px solid #0084ff;
        transition: all 0.3s ease;
    }
    
    .llm-card:hover {
        box-shadow: 0 12px 35px rgba(0, 132, 255, 0.4);
        transform: translateX(5px);
        border-left-color: #00d4ff;
    }
    
    /* Dark LLM Card */
    .dark-llm-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border: 1px solid #2d3746;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.6);
        transition: all 0.3s ease;
    }
    
    .dark-llm-card:hover {
        box-shadow: 0 12px 35px rgba(0, 212, 255, 0.3);
        transform: translateX(5px);
        border-color: #00d4ff;
    }
    
    /* Stats Highlight */
    .stats-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stats-highlight h3 {
        color: white;
        font-size: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Dark Section Background */
    .dark-section {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        padding: 3rem 2rem;
        border-radius: 25px;
        margin: 3rem 0;
        border: 1px solid #2d3746;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.6);
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
    
    @keyframes glow {
        0%, 100% {
            box-shadow: 0 0 5px rgba(0, 212, 255, 0.3);
        }
        50% {
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.6);
        }
    }
    
    /* Button Overrides - DARK THEME */
    .stButton > button {
        background: linear-gradient(135deg, #0084ff 0%, #00d4ff 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 132, 255, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 132, 255, 0.6);
        background: linear-gradient(135deg, #00d4ff 0%, #0084ff 100%);
    }
    
    /* Selectbox Styling - DARK THEME */
    .stSelectbox > div > div {
        border-radius: 12px;
        border: 2px solid #2d3746;
        font-size: 1.1rem;
        background: #1a1a2e;
        color: white;
    }
    
    /* Dataframe Styling - DARK THEME */
    .dataframe {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.6);
        background: #1a1a2e !important;
    }
    
    /* Streamlit component overrides for dark theme */
    .stDataFrame {
        background: #1a1a2e !important;
    }
    
    /* Text color overrides */
    .stMarkdown, .stText, .stAlert, .stSuccess, .stWarning, .stError {
        color: #ffffff !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(135deg, #0084ff 0%, #00d4ff 100%);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    
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
# VISUALIZATION FUNCTIONS FOR DARK THEME
# =============================================

def display_anomaly_results(viz_data, fraud_df):
    """Display anomaly detection results in dark theme"""
    
    # Top Job Categories
    if 'job_categories' in viz_data:
        st.markdown('<h3 class="dark-metric-label">👔 Top Job Categories involved in Fraud</h3>', unsafe_allow_html=True)
        job_data = viz_data['job_categories']
        job_df = pd.DataFrame({
            'Job Category': [str(key).replace('JOBctg_', '').replace('_', ' ') for key in job_data.keys()],
            'Cases': list(job_data.values())
        }).sort_values('Cases', ascending=False).head(5)
        
        cols = st.columns(5)
        for idx, (_, row) in enumerate(job_df.iterrows()):
            with cols[idx]:
                st.markdown(f"""
                <div class="dark-metric-card">
                    <div class="dark-metric-label">{row['Job Category']}</div>
                    <div class="dark-metric-value">{row['Cases']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Age Groups
    if 'age_groups' in viz_data:
        st.markdown('<h3 class="dark-metric-label">👥 Top Age Groups involved in Fraud</h3>', unsafe_allow_html=True)
        age_data = viz_data['age_groups']
        age_df = pd.DataFrame({
            'Age Group': [str(key).replace('dob_', '').upper() for key in age_data.keys()],
            'Cases': list(age_data.values())
        }).sort_values('Cases', ascending=False)
        
        cols = st.columns(len(age_df))
        for idx, (_, row) in enumerate(age_df.iterrows()):
            with cols[idx]:
                st.markdown(f"""
                <div class="dark-metric-card">
                    <div class="dark-metric-label">{row['Age Group']}</div>
                    <div class="dark-metric-value">{row['Cases']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Transaction Categories
    if 'transaction_categories' in viz_data:
        st.markdown('<h3 class="dark-metric-label">🛒 Top Transaction Categories involved in Fraud</h3>', unsafe_allow_html=True)
        txn_data = viz_data['transaction_categories']
        txn_df = pd.DataFrame({
            'Category': [str(key).replace('TXNctg_', '').replace('_', ' ') for key in txn_data.keys()],
            'Cases': list(txn_data.values())
        }).sort_values('Cases', ascending=False).head(4)
        
        cols = st.columns(4)
        for idx, (_, row) in enumerate(txn_df.iterrows()):
            with cols[idx]:
                st.markdown(f"""
                <div class="dark-metric-card">
                    <div class="dark-metric-label">{row['Category']}</div>
                    <div class="dark-metric-value">{row['Cases']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Amount Analysis
    if 'amount_analysis' in viz_data:
        st.markdown('<h3 class="dark-metric-label">💰 Amount Analysis</h3>', unsafe_allow_html=True)
        amt_data = viz_data['amount_analysis']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="dark-metric-card">
                <div class="dark-metric-label">Normal Avg</div>
                <div class="dark-metric-value">${amt_data.get('normal_avg', 0):.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="dark-metric-card">
                <div class="dark-metric-label">Fraud Avg</div>
                <div class="dark-metric-value">${amt_data.get('fraud_avg', 0):.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            increase_pct = amt_data.get('increase_pct', 0)
            st.markdown(f"""
            <div class="dark-metric-card">
                <div class="dark-metric-label">Amount Hike</div>
                <div class="dark-metric-value" style="color: {'#ff6b6b' if increase_pct > 0 else '#00d4ff'}">{increase_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

# =============================================
# MAIN APP
# =============================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🏦 TruLedger.AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">An Explainable AI Prototype for ML-powered Financial Fraud Detection</p>', unsafe_allow_html=True)
    
    
    # Dataset Selection
    st.markdown('<h2 class="section-header">📁 Upload Documents</h2>', unsafe_allow_html=True)
    
    dataset_options = {
        "TransactionLogs-1",
        "TransactionLogs-2",
        "TransactionLogs-3"
    }
    
    selected_key = st.selectbox(
        "💡 Upload anything from Credit Card Transactions to Bank's Finance Records. :)" \
        "⚠️ This is a prototype. Please upload from pre-uploaded datasets. Why? :- Due to potential data heterogenity conflicts.",
        options=list(dataset_options),
        help="Choose a pre-uploaded dataset to run the pipeline on."
    )
    
    # Run Pipeline Button
    if st.button("📌 Upload", type="primary"):
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
            st.success(f"✅ Successfully ran Fraud Detection Pipeline from Data Processing & Visualization to XAI-LLM Explanations")
            
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
        
        col1, col2, col3 = st.columns(3)
        
        metrics = [
            ("Precision", model_perf.get('precision', 90)),
            ("Recall", model_perf.get('recall', 75)),
            ("F1-Score", model_perf.get('f1_score', 0.81))
        ]
        
        for col, (label, value) in zip([col1, col2, col3], metrics):
            # percentages for all 3 except for f1_score
            with col:
                if label in ["Precision", "Recall"]:
                    display_value = f"{value:.1%}"
                    bar_width = f"{value:.1%}"
                else:
                    display_value = f"{value:.2f}"
                    bar_width = f"{value * 100:.1f}%"

                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{display_value}</div>
                    <div class="metric-bar">
                        <div class="metric-bar-fill" style="width: {bar_width};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Stats Highlight
        st.markdown(f"""
        <div class="stats-highlight">
            <h3>🎯 Exceptional Detection Performance</h3>
            <h3>Detected {len(fraud_df)} fraudulent transactions.</h3>
            <p>The XGBoost model achieves industrial-level fraud detection with minimal false claims</p>
        </div>
        """, unsafe_allow_html=True)
        
        # ML ANOMALY RESULTS SECTION - DARK THEME
        st.markdown('<h2 class="dark-section-header">📊 ML Anomaly Detection Results</h2>', unsafe_allow_html=True)
        
        fraud_df = st.session_state.get('fraud_df', pd.DataFrame())
        viz_data = st.session_state.get('viz_data', {})
        
        if not fraud_df.empty:
            # Display anomaly results in dark theme
            display_anomaly_results(viz_data, fraud_df)
            
            # Fraud Transactions Table in Dark Theme
            st.markdown('<h3 class="dark-metric-label" style="margin-top: 2rem;">🔍 Detected Fraud Transactions</h3>', unsafe_allow_html=True)
            
            # Style the dataframe for dark theme
            styled_df = fraud_df.head(10).style.set_properties(**{
                'background-color': '#1a1a2e',
                'color': 'white',
                'border-color': '#2d3746'
            })
            
            st.dataframe(styled_df, use_container_width=True, height=400)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # LLM Explanations in Dark Theme
        if st.session_state.get('llm_explanations'):
            st.markdown('<h2 class="section-header">🧠 AI-Powered Fraud Explanations</h2>', unsafe_allow_html=True)
            
            explanations = st.session_state.get('llm_explanations', [])
            current_page = st.session_state.get('current_explanation_page', 0)
            items_per_page = 6
            
            # Calculate pagination
            start_idx = current_page * items_per_page
            end_idx = start_idx + items_per_page
            current_explanations = explanations[start_idx:end_idx]
            total_pages = (len(explanations) + items_per_page - 1) // items_per_page
            
            # Display current page explanations
            for i, explanation in enumerate(current_explanations):
                transaction_id = explanation.get('transaction_id', i + start_idx + 1)
                risk_factors = explanation.get('risk_factors', [])
                confidence = explanation.get('confidence', 'medium')
                explanation_text = explanation.get('explanation', 'No explanation available')
                
                confidence_color = {'high': '#ff6b6b', 'medium': '#f39c12', 'low': '#27ae60'}.get(confidence, '#64748b')
                
                st.markdown(f"""
                <div class="dark-llm-card">
                    <h4 style="color: #00d4ff; margin-bottom: 1rem;">🚨 Transaction #{transaction_id}</h4>
                    <p><strong style="color: #94a3b8;">Confidence:</strong> <span style="color: {confidence_color}; font-weight: 700;">{confidence.upper()}</span></p>
                    <p><strong style="color: #94a3b8;">Risk Factors:</strong> <span style="color: white;">{', '.join(risk_factors)}</span></p>
                    <p><strong style="color: #94a3b8;">AI Explanation:</strong></p>
                    <p style="color: #e2e8f0; background: rgba(0, 212, 255, 0.1); padding: 1rem; border-radius: 8px; border-left: 3px solid #00d4ff;">{explanation_text}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Pagination controls
            if total_pages > 1:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    if current_page > 0:
                        if st.button("⬅️ Previous", key="prev_llm"):
                            st.session_state.current_explanation_page -= 1
                            st.rerun()
                with col2:
                    st.markdown(f'<p style="text-align: center; color: #94a3b8;">Page {current_page + 1} of {total_pages}</p>', unsafe_allow_html=True)
                with col3:
                    if end_idx < len(explanations):
                        if st.button("Next ➡️", key="next_llm"):
                            st.session_state.current_explanation_page += 1
                            st.rerun()

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
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #94a3b8; padding: 2rem; font-size: 0.9rem;'>
        <p style='font-weight: 600; font-size: 1.1rem; color: #00d4ff;'>Developed by Yashwanth Krishna Devanaboina</p>
        <p>AI/ML Engineer | Software Developer | CS student at Lnu | AWS Certified Cloud Practitioner | Cisco Certified Data Analyst</p>
        <p style='margin-top: 1rem;'>© 2024 TruLedger.AI | Enterprise-Grade Fraud Detection</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()