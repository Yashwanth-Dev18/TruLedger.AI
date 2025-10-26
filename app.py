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
# 🎨 CUSTOM CSS STYLING
# =============================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .tech-badge {
        background-color: #e1f5fe;
        padding: 8px 16px;
        border-radius: 20px;
        margin: 5px;
        display: inline-block;
        font-weight: 500;
    }
    .fraud-card {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .processing-step {
        background-color: #e8f5e8;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 4px solid #4caf50;
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
# 📊 STREAMLIT CHART FUNCTIONS
# =============================================

def display_bar_chart(data, title):
    """Display a bar chart using Streamlit elements"""
    if not data or len(data) == 0:
        st.info("No data available for chart")
        return
    
    st.subheader(title)
    
    # Convert to DataFrame for display
    chart_data = pd.DataFrame({
        'Category': [str(key).replace('JOBctg_', '').replace('_', ' ') for key in data.keys()],
        'Count': list(data.values())
    })
    
    # Display as bar chart
    if not chart_data.empty:
        st.bar_chart(chart_data.set_index('Category'))
        st.dataframe(chart_data, use_container_width=True)

def display_pie_chart(data, title):
    """Display a pie chart using Streamlit progress bars"""
    if not data or len(data) == 0:
        st.info("No data available for chart")
        return
    
    st.subheader(title)
    
    total = sum(data.values()) if data.values() else 1
    for category, count in data.items():
        percentage = (count / total) * 100
        clean_category = str(category).replace('dob_', '').upper()
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write(f"**{clean_category}**")
        with col2:
            st.progress(float(percentage/100))
        with col3:
            st.write(f"{count} cases")

# =============================================
# 🎯 MAIN APP FUNCTION
# =============================================

def main():
    # Check environment first
    env_ok = check_environment()
    
    # =============================================
    # 🏦 HEADER SECTION
    # =============================================
    
    st.markdown('<h1 class="main-header">🏦 TruLedger</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">An Explainable AI Prototype for ML-powered Fraud Detection in Finance Records</p>', unsafe_allow_html=True)
    
    # =============================================
    # 🔧 TECHNOLOGIES SECTION
    # =============================================
    
    st.markdown("---")
    st.header("🛠️ Technologies Involved")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="tech-badge">PySpark</div>', unsafe_allow_html=True)
        st.markdown('<div class="tech-badge">XGBoost</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="tech-badge">SHAP</div>', unsafe_allow_html=True)
        st.markdown('<div class="tech-badge">LLM (Groq)</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="tech-badge">Streamlit</div>', unsafe_allow_html=True)
        st.markdown('<div class="tech-badge">Data Visualization</div>', unsafe_allow_html=True)
        
    with col4:
        st.markdown('<div class="tech-badge">Feature Engineering</div>', unsafe_allow_html=True)
        st.markdown('<div class="tech-badge">Anomaly Detection</div>', unsafe_allow_html=True)
    
    # =============================================
    # 📁 DATASET SELECTION & PROCESSING SECTION
    # =============================================
    
    st.markdown("---")
    st.header("📁 Dataset Selection & Analysis")
    
    # Dataset options
    dataset_options = {
        "TransactionLogs-1": "Small business transactions (1,000 records)",
        "TransactionLogs-2": "Medium enterprise transactions (5,000 records)", 
        "TransactionLogs-3": "Large financial logs (10,000 records)"
    }
    
    selected_dataset = st.selectbox(
        "Choose a dataset to analyze:",
        options=list(dataset_options.keys()),
        help="Select a dataset to run the complete fraud detection pipeline"
    )
    
    st.info(f"**{selected_dataset}**: {dataset_options[selected_dataset]}")
    
    if st.button("🚀 Run Complete Fraud Detection Pipeline", type="primary"):
        if not env_ok:
            st.error("❌ Please install required packages first.")
            return
            
        # Create progress tracking
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
            st.success(f"✅ Pipeline complete! Found {len(fraud_df)} suspicious transactions.")
            st.session_state.analysis_complete = True
            st.session_state.fraud_df = fraud_df
            st.session_state.viz_data = viz_data
            st.session_state.selected_dataset = selected_dataset
        else:
            st.warning("⚠️ Analysis completed but no fraud patterns detected in this dataset.")
            st.session_state.analysis_complete = True
    
    # =============================================
    # 📊 RESULTS DISPLAY SECTION
    # =============================================
    
    if st.session_state.get('analysis_complete', False):
        st.markdown("---")
        st.header("📊 Analysis Results")
        
        # Show which dataset is being analyzed
        if st.session_state.get('selected_dataset'):
            st.info(f"📁 Currently analyzing: **{st.session_state.selected_dataset}**")
        
        viz_data = st.session_state.get('viz_data', {})
        fraud_df = st.session_state.get('fraud_df', pd.DataFrame())
        
        if not fraud_df.empty:
            # Display fraud metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Fraud Cases Detected", len(fraud_df))
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                avg_amount = fraud_df.get('amt', fraud_df.get('transaction_amount', 0)).mean()
                st.metric("Average Fraud Amount", f"${avg_amount:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                avg_confidence = fraud_df.get('fraud_probability', 0).mean()
                st.metric("Average Confidence", f"{avg_confidence:.1%}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                total_txns = viz_data.get('model_performance', {}).get('total_transactions', 0)
                detection_rate = viz_data.get('model_performance', {}).get('detection_rate', 0)
                st.metric("Detection Rate", f"{detection_rate:.2%}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Display charts
            col1, col2 = st.columns(2)
            
            with col1:
                if 'job_categories' in viz_data:
                    display_bar_chart(
                        viz_data['job_categories'],
                        "👔 Job Categories in Fraud"
                    )
                
                if 'amount_analysis' in viz_data:
                    amt_data = viz_data['amount_analysis']
                    st.subheader("💰 Amount Analysis")
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
                        "👥 Age Groups in Fraud"
                    )
                
                if 'transaction_categories' in viz_data:
                    st.subheader("🛒 Transaction Categories")
                    txn_data = viz_data['transaction_categories']
                    for category, count in txn_data.items():
                        clean_category = str(category).replace('TXNctg_', '').replace('_', ' ')
                        st.write(f"**{clean_category}**: {count} cases")
            
            # Display fraud transactions table
            st.subheader("🔍 Detected Fraud Transactions")
            display_columns = ['transaction_amount', 'transaction_hour', 'fraud_probability']
            available_columns = [col for col in display_columns if col in fraud_df.columns]
            
            if available_columns:
                st.dataframe(fraud_df[available_columns].head(20), use_container_width=True)
        
        else:
            st.info("🎉 No fraudulent transactions detected in this dataset!")
    
    # =============================================
    # 📝 FOOTER
    # =============================================
    
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>🔒 <strong>TruLedger</strong> - Explainable AI for Financial Security</p>
            <p>Built with PySpark, XGBoost, SHAP, and LLM technologies</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()