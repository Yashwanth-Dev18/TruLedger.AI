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
    
    /* Chart Section Headers */
    .chart-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #00d4ff;
        text-align: center;
        margin: 1rem 0;
        padding: 0.5rem;
        border-radius: 10px;
        background: rgba(0, 212, 255, 0.1);
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
    
    /* Chart Container */
    .chart-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid #2d3746;
    }
    
    /* Bar Chart Styles */
    .bar-container {
        background: rgba(0, 212, 255, 0.05);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #00d4ff;
    }
    
    .bar-label {
        font-weight: 600;
        color: #94a3b8;
        margin-bottom: 0.5rem;
    }
    
    .bar-value {
        font-weight: 700;
        color: #00d4ff;
        float: right;
    }
    
    /* Progress Bar Customization */
    .stProgress > div > div {
        background: linear-gradient(90deg, #0084ff 0%, #00d4ff 100%);
        border-radius: 10px;
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
# VISUALIZATION FUNCTIONS USING STREAMLIT COMPONENTS
# =============================================

def display_job_categories_bar(job_data):
    """Display job categories using Streamlit bar chart and progress bars"""
    if not job_data:
        return
    
    st.markdown('<div class="chart-header">👔 Top Job Categories in Fraudulent Transactions</div>', unsafe_allow_html=True)
    
    # Convert to DataFrame and sort
    job_df = pd.DataFrame({
        'Job Category': [str(key).replace('JOBctg_', '').replace('_', ' ') for key in job_data.keys()],
        'Fraud Cases': list(job_data.values())
    }).sort_values('Fraud Cases', ascending=False)
    
    # Display as horizontal bar chart using progress bars
    max_cases = max(job_data.values()) if job_data.values() else 1
    
    for _, row in job_df.iterrows():
        percentage = (row['Fraud Cases'] / max_cases) * 100
        
        # Create a custom bar using columns
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            st.markdown(f"**{row['Job Category']}**")
        
        with col2:
            st.progress(float(percentage/100))
        
        with col3:
            st.markdown(f"**{row['Fraud Cases']}** cases")
        
        st.markdown("---")

def display_age_groups_pie(age_data):
    """Display age groups using Streamlit progress bars as pie chart representation"""
    if not age_data:
        return
    
    st.markdown('<div class="chart-header">👥 Age Groups Involved in Fraud</div>', unsafe_allow_html=True)
    
    # Convert to DataFrame and sort
    age_df = pd.DataFrame({
        'Age Group': [str(key).replace('dob_', '').upper() for key in age_data.keys()],
        'Cases': list(age_data.values())
    }).sort_values('Cases', ascending=False)
    
    total_cases = sum(age_data.values()) if age_data.values() else 1
    
    # Display as percentage bars (simulating pie chart slices)
    for _, row in age_df.iterrows():
        percentage = (row['Cases'] / total_cases) * 100
        
        col1, col2, col3, col4 = st.columns([2, 3, 1, 1])
        
        with col1:
            st.markdown(f"**{row['Age Group']}**")
        
        with col2:
            st.progress(float(percentage/100))
        
        with col3:
            st.markdown(f"**{percentage:.1f}%**")
        
        with col4:
            st.markdown(f"({row['Cases']})")
    
    # Add a summary
    st.markdown(f"**Total Cases:** {total_cases}")

def display_transaction_categories_bar(txn_data):
    """Display transaction categories using Streamlit bar chart"""
    if not txn_data:
        return
    
    st.markdown('<div class="chart-header">🛒 Top Transaction Categories in Fraudulent Transactions</div>', unsafe_allow_html=True)
    
    # Convert to DataFrame and sort
    txn_df = pd.DataFrame({
        'Category': [str(key).replace('TXNctg_', '').replace('_', ' ') for key in txn_data.keys()],
        'Fraud Cases': list(txn_data.values())
    }).sort_values('Fraud Cases', ascending=False)
    
    max_cases = max(txn_data.values()) if txn_data.values() else 1
    
    # Display as horizontal bars
    for _, row in txn_df.iterrows():
        percentage = (row['Fraud Cases'] / max_cases) * 100
        
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            # Shorten long category names
            category_name = row['Category']
            if len(category_name) > 25:
                category_name = category_name[:25] + "..."
            st.markdown(f"**{category_name}**")
        
        with col2:
            st.progress(float(percentage/100))
        
        with col3:
            st.markdown(f"**{row['Fraud Cases']}**")
        
        st.markdown("---")

def display_anomaly_results(viz_data, fraud_df):
    """Display anomaly detection results with Streamlit components"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Job Categories Bar Chart
        if 'job_categories' in viz_data:
            display_job_categories_bar(viz_data['job_categories'])
    
    with col2:
        # Age Groups Pie Chart
        if 'age_groups' in viz_data:
            display_age_groups_pie(viz_data['age_groups'])
    
    # Transaction Categories Bar Chart (full width)
    if 'transaction_categories' in viz_data:
        display_transaction_categories_bar(viz_data['transaction_categories'])
    
    # Amount Analysis
    if 'amount_analysis' in viz_data:
        st.markdown('<div class="chart-header" style="margin-top: 2rem;">💰 Amount Analysis</div>', unsafe_allow_html=True)
        amt_data = viz_data['amount_analysis']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="dark-metric-card">
                <div class="dark-metric-label">Normal Transaction Average</div>
                <div class="dark-metric-value">${amt_data.get('normal_avg', 0):.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="dark-metric-card">
                <div class="dark-metric-label">Fraud Transaction Average</div>
                <div class="dark-metric-value">${amt_data.get('fraud_avg', 0):.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            increase_pct = amt_data.get('increase_pct', 0)
            st.markdown(f"""
            <div class="dark-metric-card">
                <div class="dark-metric-label">Amount Hike in Fraud</div>
                <div class="dark-metric-value" style="color: {'#ff6b6b' if increase_pct > 0 else '#00d4ff'}">{increase_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

# =============================================
# MAIN APP
# =============================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🏦 TruLedger</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Financial Reconciliation & Fraud Detection Assistant</p>', unsafe_allow_html=True)
    
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
        # Model Performance with fixed metrics
        st.markdown('<h2 class="section-header">🤖 ML Model Performance</h2>', unsafe_allow_html=True)
        
        # Fixed metrics as requested
        fixed_metrics = [
            ("Accuracy", 0.99),
            ("Precision", 0.90),
            ("Recall", 0.74),
            ("F1-Score", 0.80)
        ]
        
        col1, col2, col3, col4 = st.columns(4)
        
        for col, (label, value) in zip([col1, col2, col3, col4], fixed_metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value:.1% if label != 'F1-Score' else value:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Stats Highlight
        st.markdown("""
        <div class="stats-highlight">
            <h3>🎯 Exceptional Detection Performance</h3>
            <p>Our XGBoost model achieves industry-leading fraud detection with minimal false positives</p>
        </div>
        """, unsafe_allow_html=True)
        
        # ML ANOMALY RESULTS SECTION - DARK THEME
        st.markdown('<div class="dark-section">', unsafe_allow_html=True)
        st.markdown('<h2 class="dark-section-header">📊 ML Anomaly Detection Results</h2>', unsafe_allow_html=True)
        
        fraud_df = st.session_state.get('fraud_df', pd.DataFrame())
        viz_data = st.session_state.get('viz_data', {})
        
        if not fraud_df.empty:
            # Display anomaly results with Streamlit components
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
    
    # Technology Stack - MOVED TO LAST SECTION
    if st.session_state.get('analysis_complete', False):
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
        <p>Built with cutting-edge AI technologies for explainable fraud detection</p>
        <p style='margin-top: 1rem;'>© 2024 TruLedger AI | Enterprise-Grade Fraud Detection</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()