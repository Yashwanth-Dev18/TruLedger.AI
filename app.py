import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys

# Import our processing functions
from dataProcessor import process_transaction_data
from ML_Engine import run_fraud_detection

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'page' not in st.session_state:
    st.session_state.page = 0
if 'selected_dataset' not in st.session_state:
    st.session_state.selected_dataset = None
if 'fraud_df' not in st.session_state:
    st.session_state.fraud_df = pd.DataFrame()
if 'viz_data' not in st.session_state:
    st.session_state.viz_data = {}

def check_environment():
    """Check if required dependencies are available"""
    missing_packages = []
    
    try:
        import pandas
    except ImportError:
        missing_packages.append('pandas')
    
    try:
        import sklearn
    except ImportError:
        missing_packages.append('scikit-learn')
    
    try:
        import joblib
    except ImportError:
        missing_packages.append('joblib')
    
    if missing_packages:
        st.warning(f"⚠️ Missing packages: {', '.join(missing_packages)}")
        st.info("Some features may not work properly. Please install required packages.")
        return False
    return True

# =============================================
# 🎯 PAGE CONFIGURATION
# =============================================

st.set_page_config(
    page_title="TruLedger - AI Fraud Detection",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# 🎨 ENHANCED CSS STYLING
# =============================================

st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #2E4057;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2E4057, #4A6FA5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        font-size: 1.3rem;
        color: #6C757D;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    .tech-badge {
        background: linear-gradient(135deg, #F8F9FA, #E9ECEF);
        color: #495057;
        padding: 10px 18px;
        border-radius: 25px;
        margin: 6px;
        display: inline-block;
        font-weight: 500;
        border: 1px solid #DEE2E6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    .tech-badge:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .fraud-card {
        background: linear-gradient(135deg, #FFF5F5, #FFE8E8);
        border-left: 5px solid #E74C3C;
        padding: 20px;
        border-radius: 12px;
        margin: 12px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .metric-card {
        background: linear-gradient(135deg, #FFFFFF, #F8F9FA);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        border: 1px solid #E9ECEF;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    .processing-step {
        background: linear-gradient(135deg, #F0F9FF, #E6F7FF);
        padding: 15px;
        border-radius: 10px;
        margin: 8px 0;
        border-left: 5px solid #3498DB;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .success-box {
        background: linear-gradient(135deg, #F0FFF4, #E6FFFA);
        border-left: 5px solid #38B2AC;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #FFFBF0, #FFF9E6);
        border-left: 5px solid #ED8936;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
    }
    .section-header {
        color: #2D3748;
        font-weight: 600;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E2E8F0;
    }
    .dataset-card {
        background: linear-gradient(135deg, #FFFFFF, #F7FAFC);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    .dataset-card:hover {
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    .stButton button {
        background: linear-gradient(135deg, #3498DB, #2980B9);
        color: white;
        border: none;
        padding: 12px 30px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 25px;
        transition: all 0.3s ease;
        font-weight: 600;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #2980B9, #2471A3);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(52, 152, 219, 0.3);
    }
    .footer {
        background: linear-gradient(135deg, #2E4057, #34495E);
        color: white;
        padding: 30px;
        border-radius: 15px;
        margin-top: 40px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# 📊 DATA LOADING FUNCTIONS
# =============================================

@st.cache_data
def load_fraud_data():
    """Load detected fraud transactions"""
    try:
        if os.path.exists('detected_fraud_transactions.csv'):
            fraud_df = pd.read_csv('detected_fraud_transactions.csv')
            return fraud_df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading fraud data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_visualization_data():
    """Load pre-computed visualization data"""
    try:
        if os.path.exists('app_visualization_data.json'):
            with open('app_visualization_data.json', 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        st.error(f"Error loading visualization data: {e}")
        return None

# =============================================
# 📊 ENHANCED CHART FUNCTIONS
# =============================================

def display_bar_chart(data, title):
    """Display a bar chart using Streamlit elements"""
    if not data or len(data) == 0:
        st.info("📊 No data available for chart")
        return
    
    st.subheader(title)
    
    # Convert to DataFrame for display
    chart_data = pd.DataFrame({
        'Category': [str(key).replace('JOBctg_', '').replace('_', ' ') for key in data.keys()],
        'Count': list(data.values())
    })
    
    # Display as bar chart
    if not chart_data.empty:
        st.bar_chart(chart_data.set_index('Category'), height=400)
        with st.expander("📋 View Data Table"):
            st.dataframe(chart_data, use_container_width=True)

def display_pie_chart(data, title):
    """Display a pie chart using Streamlit progress bars"""
    if not data or len(data) == 0:
        st.info("📊 No data available for chart")
        return
    
    st.subheader(title)
    
    total = sum(data.values()) if data.values() else 1
    for category, count in data.items():
        percentage = (count / total) * 100
        clean_category = str(category).replace('dob_', '').replace('_', ' ').upper()
        
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.write(f"**{clean_category}**")
        with col2:
            st.progress(float(percentage/100))
        with col3:
            st.write(f"{count} ({percentage:.1f}%)")

# =============================================
# 🎯 ENHANCED MAIN APP FUNCTION
# =============================================

def main():
    # Check environment first
    env_ok = check_environment()
    
    # =============================================
    # 🏦 ENHANCED HEADER SECTION
    # =============================================
    
    col_header1, col_header2, col_header3 = st.columns([1, 2, 1])
    with col_header2:
        st.markdown('<h1 class="main-header">🏦 TruLedger</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Advanced AI-Powered Financial Fraud Detection & Analysis</p>', unsafe_allow_html=True)
    
    # =============================================
    # 🔧 ENHANCED TECHNOLOGIES SECTION
    # =============================================
    
    st.markdown("---")
    st.markdown('<h2 class="section-header">🛠️ Core Technologies</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="tech-badge">🔍 PySpark</div>', unsafe_allow_html=True)
        st.markdown('<div class="tech-badge">🌳 XGBoost</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="tech-badge">📊 SHAP</div>', unsafe_allow_html=True)
        st.markdown('<div class="tech-badge">🤖 LLM (Groq)</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="tech-badge">🚀 Streamlit</div>', unsafe_allow_html=True)
        st.markdown('<div class="tech-badge">📈 Data Visualization</div>', unsafe_allow_html=True)
        
    with col4:
        st.markdown('<div class="tech-badge">⚙️ Feature Engineering</div>', unsafe_allow_html=True)
        st.markdown('<div class="tech-badge">🎯 Anomaly Detection</div>', unsafe_allow_html=True)
    
    # =============================================
    # 📁 ENHANCED DATASET SELECTION SECTION
    # =============================================
    
    st.markdown("---")
    st.markdown('<h2 class="section-header">📁 Dataset Selection & Analysis</h2>', unsafe_allow_html=True)
    
    # Enhanced dataset options with icons
    dataset_options = {
        "TransactionLogs-1": {"desc": "Small business transactions (1,000 records)", "icon": "📊"},
        "TransactionLogs-2": {"desc": "Medium enterprise transactions (5,000 records)", "icon": "📈"}, 
        "TransactionLogs-3": {"desc": "Large financial logs (10,000 records)", "icon": "🏢"}
    }
    
    selected_dataset = st.selectbox(
        "🎯 Choose a dataset to analyze:",
        options=list(dataset_options.keys()),
        help="Select a dataset to run the complete fraud detection pipeline",
        format_func=lambda x: f"{dataset_options[x]['icon']} {x} - {dataset_options[x]['desc']}"
    )
    
    # Display selected dataset info in a card
    st.markdown(f"""
    <div class="dataset-card">
        <h4>📋 Selected Dataset</h4>
        <p><strong>{selected_dataset}</strong>: {dataset_options[selected_dataset]['desc']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced button with better styling
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("🚀 Run Complete Fraud Detection Pipeline", type="primary", use_container_width=True):
            if not env_ok:
                st.error("❌ Please install required packages first.")
                return
                
            # Create enhanced progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Check if dataset exists
            input_file = os.path.join("Uploaded_Datasets", "Raw", f"{selected_dataset}.csv")
            if not os.path.exists(input_file):
                st.error(f"❌ Dataset file not found: {input_file}")
                st.info("Please make sure your dataset files are in the 'Uploaded_Datasets/Raw/' directory")
                return
            
            # Step 1: Data Processing
            status_text.markdown('<div class="processing-step">🔄 Step 1/3: Processing raw data...</div>', unsafe_allow_html=True)
            progress_bar.progress(25)
            
            processed_file = process_transaction_data(input_file)
            
            if processed_file is None:
                st.error("❌ Failed to process data. Please check the file path and try again.")
                return
            
            # Step 2: Fraud Detection
            status_text.markdown('<div class="processing-step">🔍 Step 2/3: Running fraud detection...</div>', unsafe_allow_html=True)
            progress_bar.progress(60)
            
            success = run_fraud_detection(processed_file)
            
            if not success:
                st.error("❌ Failed to run fraud detection.")
                return
                
            # Step 3: Load Results
            status_text.markdown('<div class="processing-step">📊 Step 3/3: Loading results...</div>', unsafe_allow_html=True)
            progress_bar.progress(90)
            
            fraud_df = load_fraud_data()
            viz_data = load_visualization_data()
            
            progress_bar.progress(100)
            status_text.markdown('<div class="processing-step">✅ Analysis complete!</div>', unsafe_allow_html=True)
            
            if not fraud_df.empty and viz_data is not None:
                st.markdown(f'<div class="success-box"><h4>✅ Pipeline Complete!</h4><p>Found <strong>{len(fraud_df)}</strong> suspicious transactions requiring review.</p></div>', unsafe_allow_html=True)
                st.session_state.analysis_complete = True
                st.session_state.fraud_df = fraud_df
                st.session_state.viz_data = viz_data
                st.session_state.selected_dataset = selected_dataset
            else:
                st.markdown('<div class="warning-box"><h4>⚠️ Analysis Complete</h4><p>No fraud patterns detected in this dataset. All transactions appear normal.</p></div>', unsafe_allow_html=True)
                st.session_state.analysis_complete = True
    
    # =============================================
    # 📊 ENHANCED RESULTS DISPLAY SECTION
    # =============================================
    
    if st.session_state.get('analysis_complete', False):
        st.markdown("---")
        st.markdown('<h2 class="section-header">📊 Analysis Results</h2>', unsafe_allow_html=True)
        
        # Show which dataset is being analyzed
        if st.session_state.get('selected_dataset'):
            st.info(f"📁 Currently analyzing: **{st.session_state.selected_dataset}**")
        
        viz_data = st.session_state.get('viz_data', {})
        fraud_df = st.session_state.get('fraud_df', pd.DataFrame())
        
        if not fraud_df.empty:
            # Enhanced fraud metrics with better styling
            st.subheader("📈 Key Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("🚨 Fraud Cases", len(fraud_df))
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                avg_amount = fraud_df.get('amt', fraud_df.get('transaction_amount', 0)).mean()
                st.metric("💰 Avg Fraud Amount", f"${avg_amount:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                avg_confidence = fraud_df.get('fraud_probability', 0).mean()
                st.metric("🎯 Avg Confidence", f"{avg_confidence:.1%}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                total_txns = viz_data.get('model_performance', {}).get('total_transactions', 0)
                detection_rate = viz_data.get('model_performance', {}).get('detection_rate', 0)
                st.metric("📊 Detection Rate", f"{detection_rate:.2%}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Enhanced charts section
            st.subheader("📊 Fraud Patterns & Insights")
            col1, col2 = st.columns(2)
            
            with col1:
                if 'job_categories' in viz_data:
                    display_bar_chart(
                        viz_data['job_categories'],
                        "👔 Job Categories in Fraud"
                    )
                
                if 'amount_analysis' in viz_data:
                    amt_data = viz_data['amount_analysis']
                    st.subheader("💰 Transaction Amount Analysis")
                    col_a1, col_a2, col_a3 = st.columns(3)
                    
                    with col_a1:
                        st.metric("Normal Avg", f"${amt_data.get('normal_avg', 0):.2f}")
                    
                    with col_a2:
                        st.metric("Fraud Avg", f"${amt_data.get('fraud_avg', 0):.2f}")
                    
                    with col_a3:
                        st.metric("Increase", f"+{amt_data.get('increase_pct', 0):.1f}%")
            
            with col2:
                if 'age_groups' in viz_data:
                    display_pie_chart(
                        viz_data['age_groups'],
                        "👥 Age Distribution in Fraud"
                    )
                
                if 'transaction_categories' in viz_data:
                    st.subheader("🛒 Transaction Categories")
                    txn_data = viz_data['transaction_categories']
                    for category, count in txn_data.items():
                        clean_category = str(category).replace('TXNctg_', '').replace('_', ' ').title()
                        st.write(f"**{clean_category}**: {count} cases")
            
            # Enhanced fraud transactions table
            st.subheader("🔍 Detected Fraud Transactions")
            display_columns = ['transaction_amount', 'transaction_hour', 'fraud_probability']
            available_columns = [col for col in display_columns if col in fraud_df.columns]
            
            if available_columns:
                st.dataframe(
                    fraud_df[available_columns].head(20).style.format({
                        'transaction_amount': '${:.2f}',
                        'fraud_probability': '{:.1%}'
                    }), 
                    use_container_width=True,
                    height=400
                )
        
        else:
            st.markdown("""
            <div style='text-align: center; padding: 40px; background: linear-gradient(135deg, #F0FFF4, #E6FFFA); border-radius: 15px;'>
                <h3 style='color: #38A169;'>🎉 No Fraudulent Activity Detected!</h3>
                <p style='color: #2D3748; font-size: 16px;'>All transactions in this dataset appear to be legitimate.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # =============================================
    # 📝 ENHANCED FOOTER
    # =============================================
    
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <h4 style='color: white; margin-bottom: 10px;'>🔒 TruLedger - Advanced Financial Security</h4>
        <p style='color: #CBD5E0; margin-bottom: 5px;'>Explainable AI for Real-time Fraud Detection & Prevention</p>
        <p style='color: #A0AEC0; font-size: 14px;'>Built with cutting-edge ML technologies for enterprise-grade security</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()