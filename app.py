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
if 'llm_explanations' not in st.session_state:
    st.session_state.llm_explanations = []
if 'current_explanation_page' not in st.session_state:
    st.session_state.current_explanation_page = 0

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
        font-size: 3.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0;
        font-weight: 700;
        background: linear-gradient(135deg, #1f77b4, #2e93e6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        font-size: 1.4rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    .section-header {
        font-size: 2rem;
        color: #1f77b4;
        margin: 2rem 0 1rem 0;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.5rem;
    }
    .tech-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 25px;
        margin: 8px;
        display: inline-block;
        font-weight: 600;
        font-size: 0.9rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    .tech-badge:hover {
        transform: translateY(-2px);
    }
    .fraud-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .metric-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .processing-step {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: 500;
    }
    .dataset-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #e0e0e0;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    .dataset-card:hover {
        border-color: #1f77b4;
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
    }
    .llm-explanation-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #1f77b4;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    .stats-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin: 20px 0;
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

@st.cache_data
def load_llm_explanations():
    """Load LLM explanations"""
    try:
        if os.path.exists('llm_fraud_explanations.json'):
            with open('llm_fraud_explanations.json', 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        st.error(f"Error loading LLM explanations: {e}")
        return None

# =============================================
# 📊 VISUALIZATION FUNCTIONS
# =============================================

def display_top_job_categories(job_data):
    """Display top 5 job categories involved in fraud"""
    if not job_data:
        st.info("No job category data available")
        return
    
    st.subheader("👔 Top 5 Job Categories in Fraud")
    
    # Convert to DataFrame and get top 5
    job_df = pd.DataFrame({
        'Job Category': [str(key).replace('JOBctg_', '').replace('_', ' ') for key in job_data.keys()],
        'Fraud Cases': list(job_data.values())
    }).sort_values('Fraud Cases', ascending=False).head(5)
    
    # Display as bar chart
    if not job_df.empty:
        st.bar_chart(job_df.set_index('Job Category'))
        
        # Display as metrics
        cols = st.columns(5)
        for idx, (_, row) in enumerate(job_df.iterrows()):
            with cols[idx]:
                st.metric(
                    label=row['Job Category'],
                    value=row['Fraud Cases']
                )

def display_age_groups(age_data):
    """Display age groups involved in fraud"""
    if not age_data:
        st.info("No age group data available")
        return
    
    st.subheader("👥 Age Groups in Fraud")
    
    # Convert to DataFrame
    age_df = pd.DataFrame({
        'Age Group': [str(key).replace('dob_', '').upper() for key in age_data.keys()],
        'Cases': list(age_data.values())
    }).sort_values('Cases', ascending=False)
    
    # Display pie chart using progress bars
    total = sum(age_data.values()) if age_data.values() else 1
    
    for _, row in age_df.iterrows():
        percentage = (row['Cases'] / total) * 100
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write(f"**{row['Age Group']}**")
        with col2:
            st.progress(float(percentage/100))
        with col3:
            st.write(f"{row['Cases']} cases ({percentage:.1f}%)")

def display_transaction_categories(txn_data):
    """Display transaction categories involved in fraud"""
    if not txn_data:
        st.info("No transaction category data available")
        return
    
    st.subheader("🛒 Transaction Categories in Fraud")
    
    # Convert to DataFrame and get top categories
    txn_df = pd.DataFrame({
        'Category': [str(key).replace('TXNctg_', '').replace('_', ' ') for key in txn_data.keys()],
        'Cases': list(txn_data.values())
    }).sort_values('Cases', ascending=False)
    
    # Display as metrics in columns
    cols = st.columns(4)
    for idx, (_, row) in enumerate(txn_df.iterrows()):
        if idx < 8:  # Show top 8 categories
            with cols[idx % 4]:
                st.metric(
                    label=row['Category'],
                    value=row['Cases']
                )

def display_amount_hike(amount_data):
    """Display amount hike percentage"""
    if not amount_data:
        st.info("No amount analysis data available")
        return
    
    st.subheader("💰 Amount Analysis")
    
    increase_pct = amount_data.get('increase_pct', 0)
    normal_avg = amount_data.get('normal_avg', 0)
    fraud_avg = amount_data.get('fraud_avg', 0)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Normal Transaction Average",
            value=f"${normal_avg:.2f}",
            delta="Baseline"
        )
    
    with col2:
        st.metric(
            label="Fraud Transaction Average", 
            value=f"${fraud_avg:.2f}",
            delta=f"${fraud_avg - normal_avg:.2f}"
        )
    
    with col3:
        st.metric(
            label="Amount Hike in Fraud",
            value=f"{increase_pct:.1f}%",
            delta_color="inverse"
        )

# =============================================
# 🎯 MAIN APP FUNCTION
# =============================================

def main():
    # Check environment first
    env_ok = check_environment()
    
    # =============================================
    # 🏦 HEADER SECTION
    # =============================================
    
    st.markdown('<h1 class="main-header">TruLedger.AI </h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">An Explainable AI Prototype for ML-powered Financial Fraud Detection</p>', unsafe_allow_html=True)
    
    # =============================================
    # 🔧 TECHNOLOGIES SECTION
    # =============================================
    
    st.markdown("---")
    st.markdown('<h2 class="section-header">🛠️ Advanced Technology Stack</h2>', unsafe_allow_html=True)
    
    # Tech stack in a nice grid
    tech_cols = st.columns(4)
    
    tech_stack = [
        ("PySpark", "Big Data Processing"),
        ("XGBoost", "ML Algorithm"),
        ("SHAP", "Model Explainability"), 
        ("LLM (Groq)", "AI Explanations"),
        ("Streamlit", "Web Interface"),
        ("Scikit-learn", "ML Framework"),
        ("Pandas", "Data Manipulation"),
        ("NumPy", "Numerical Computing")
    ]
    
    for idx, (tech, desc) in enumerate(tech_stack):
        with tech_cols[idx % 4]:
            st.markdown(f'<div class="tech-badge" title="{desc}">{tech}</div>', unsafe_allow_html=True)
    
    # =============================================
    # 📁 DATASET SELECTION SECTION
    # =============================================
    
    st.markdown("---")
    st.markdown('<h2 class="section-header">📁 Dataset Selection</h2>', unsafe_allow_html=True)
    
    # Dataset options with enhanced descriptions
    dataset_options = {
        "TransactionLogs-1": {
            "description": "Small business transactions dataset",
            "records": "1,000 records",
            "size": "Ideal for quick testing"
        },
        "TransactionLogs-2": {
            "description": "Medium enterprise transactions", 
            "records": "5,000 records",
            "size": "Balanced performance"
        },
        "TransactionLogs-3": {
            "description": "Large financial logs",
            "records": "10,000 records", 
            "size": "Comprehensive analysis"
        }
    }
    
    selected_key = st.selectbox(
        "🎯 Select Dataset for Analysis:",
        options=list(dataset_options.keys()),
        help="Choose a dataset to run the complete fraud detection pipeline"
    )
    
    # Display dataset info card
    if selected_key:
        dataset_info = dataset_options[selected_key]
        st.markdown(f"""
        <div class="dataset-card">
            <h3>📊 {selected_key}</h3>
            <p><strong>Description:</strong> {dataset_info['description']}</p>
            <p><strong>Records:</strong> {dataset_info['records']}</p>
            <p><strong>Use Case:</strong> {dataset_info['size']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Process button
    if st.button("🚀 Run Complete Fraud Detection Pipeline", type="primary", use_container_width=True):
        if not env_ok:
            st.error("❌ Please install required packages first.")
            return
            
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Check if dataset exists
        input_file = os.path.join("Uploaded_Datasets", "Raw", f"{selected_key}.csv")
        if not os.path.exists(input_file):
            st.error(f"❌ Dataset file not found: {input_file}")
            st.info("Please make sure your dataset files are in the 'Uploaded_Datasets/Raw/' directory")
            return
        
        # Step 1: Data Processing
        status_text.markdown('<div class="processing-step">🔄 Step 1/4: Processing raw data...</div>', unsafe_allow_html=True)
        progress_bar.progress(20)
        
        processed_file = process_transaction_data(input_file)
        
        if processed_file is None:
            st.error("❌ Failed to process data. Please check the file path and try again.")
            return
        
        # Step 2: Fraud Detection
        status_text.markdown('<div class="processing-step">🔍 Step 2/4: Running fraud detection...</div>', unsafe_allow_html=True)
        progress_bar.progress(50)
        
        success = run_fraud_detection(processed_file)
        
        if not success:
            st.error("❌ Failed to run fraud detection.")
            return
            
        # Step 3: Load Results
        status_text.markdown('<div class="processing-step">📊 Step 3/4: Loading results...</div>', unsafe_allow_html=True)
        progress_bar.progress(80)
        
        fraud_df = load_fraud_data()
        viz_data = load_visualization_data()
        llm_explanations = load_llm_explanations()
        
        progress_bar.progress(100)
        status_text.markdown('<div class="processing-step">✅ Analysis complete!</div>', unsafe_allow_html=True)
        
        if not fraud_df.empty and viz_data is not None:
            st.success(f"✅ Pipeline complete! Found {len(fraud_df)} suspicious transactions.")
            st.session_state.analysis_complete = True
            st.session_state.fraud_df = fraud_df
            st.session_state.viz_data = viz_data
            st.session_state.selected_dataset = selected_key
            st.session_state.llm_explanations = llm_explanations.get('explanations', []) if llm_explanations else []
            st.session_state.current_explanation_page = 0
        else:
            st.warning("⚠️ Analysis completed but no fraud patterns detected in this dataset.")
            st.session_state.analysis_complete = True
    
    # =============================================
    # 🤖 ML MODEL PERFORMANCE SECTION
    # =============================================
    
    if st.session_state.get('analysis_complete', False):
        st.markdown("---")
        st.markdown('<h2 class="section-header">🤖 ML Model Performance</h2>', unsafe_allow_html=True)
        
        viz_data = st.session_state.get('viz_data', {})
        model_perf = viz_data.get('model_performance', {})
        
        # Model performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Accuracy", f"{model_perf.get('accuracy', 0):.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Precision", f"{model_perf.get('precision', 0):.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Recall", f"{model_perf.get('recall', 0):.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("F1-Score", f"{model_perf.get('f1_score', 0):.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Performance highlights
        st.markdown("""
        <div class="stats-highlight">
            <h3>🎯 Model Performance Highlights</h3>
            <p>Our XGBoost model achieves exceptional fraud detection with minimal false positives</p>
        </div>
        """, unsafe_allow_html=True)
    
    # =============================================
    # 📊 ANOMALY DETECTION RESULTS SECTION
    # =============================================
    
    if st.session_state.get('analysis_complete', False):
        st.markdown("---")
        st.markdown('<h2 class="section-header">📊 Anomaly Detection Results</h2>', unsafe_allow_html=True)
        
        viz_data = st.session_state.get('viz_data', {})
        fraud_df = st.session_state.get('fraud_df', pd.DataFrame())
        
        if not fraud_df.empty:
            # Display the four main results in a 2x2 grid
            col1, col2 = st.columns(2)
            
            with col1:
                # Top Job Categories
                if 'job_categories' in viz_data:
                    display_top_job_categories(viz_data['job_categories'])
                
                # Age Groups
                if 'age_groups' in viz_data:
                    display_age_groups(viz_data['age_groups'])
            
            with col2:
                # Transaction Categories
                if 'transaction_categories' in viz_data:
                    display_transaction_categories(viz_data['transaction_categories'])
                
                # Amount Hike
                if 'amount_analysis' in viz_data:
                    display_amount_hike(viz_data['amount_analysis'])
            
            # Fraud transactions summary
            st.subheader("🔍 Fraud Transactions Overview")
            st.dataframe(fraud_df.head(10), use_container_width=True)
    
    # =============================================
    # 🧠 LLM EXPLANATIONS SECTION
    # =============================================
    
    if st.session_state.get('analysis_complete', False) and st.session_state.get('llm_explanations'):
        st.markdown("---")
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
            
            st.markdown(f"""
            <div class="llm-explanation-card">
                <h3>🚨 Fraud Transaction #{transaction_id}</h3>
                <p><strong>Confidence Level:</strong> <span style="color: {'#ff6b6b' if confidence == 'high' else '#f39c12' if confidence == 'medium' else '#27ae60'}">{confidence.upper()}</span></p>
                <p><strong>Risk Factors:</strong> {', '.join(risk_factors)}</p>
                <p><strong>AI Explanation:</strong> {explanation_text}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Pagination controls
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if current_page > 0:
                if st.button("⬅️ Previous"):
                    st.session_state.current_explanation_page -= 1
                    st.rerun()
        
        with col2:
            st.write(f"Page {current_page + 1} of {total_pages}")
        
        with col3:
            if end_idx < len(explanations):
                if st.button("Next ➡️"):
                    st.session_state.current_explanation_page += 1
                    st.rerun()
    
    # =============================================
    # 📝 FOOTER
    # =============================================
    
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <h3>🔒 TruLedger - Advanced Financial Security</h3>
            <p>Built with cutting-edge AI technologies for explainable fraud detection</p>
            <p style='font-size: 0.9rem; margin-top: 1rem;'>© 2024 TruLedger AI | Enterprise-Grade Fraud Detection</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()