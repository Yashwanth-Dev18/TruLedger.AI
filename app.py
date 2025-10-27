import streamlit as st
import pandas as pd
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
# SIMPLIFIED DARK THEME CSS
# =============================================

st.markdown("""
<style>
    .stApp {
        background: #000000;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #0084ff 0%, #00d4ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00d4ff;
        margin: 2rem 0 1rem 0;
        border-bottom: 2px solid #0084ff;
        padding-bottom: 0.5rem;
    }
    
    .chart-container {
        background: #1a1a2e;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #2d3746;
    }
    
    .metric-card {
        background: #1a1a2e;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        border-left: 4px solid #0084ff;
        margin: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00d4ff;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
    }
    
    .tech-badge {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.3rem;
        font-weight: 500;
        font-size: 0.8rem;
    }
    
    .processing-step {
        background: #4facfe;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-weight: 600;
        text-align: center;
    }
    
    .llm-card {
        background: #1a1a2e;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #0084ff;
    }
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
# SIMPLIFIED VISUALIZATION FUNCTIONS
# =============================================

def create_compact_bar_chart(data, title, emoji):
    """Create compact bar chart"""
    if not data:
        return
    
    df = pd.DataFrame({
        'Category': [str(key).replace('JOBctg_', '').replace('TXNctg_', '').replace('_', ' ').title() 
                    for key in data.keys()],
        'Count': list(data.values())
    }).sort_values('Count', ascending=False).head(5)
    
    st.markdown(f'<div class="chart-container">', unsafe_allow_html=True)
    st.markdown(f'<h4 style="color: #00d4ff; text-align: center;">{emoji} {title}</h4>', unsafe_allow_html=True)
    
    # Compact bar chart
    st.bar_chart(df.set_index('Category')['Count'], height=250)
    
    # Compact data display
    st.dataframe(df, use_container_width=True, hide_index=True, height=150)
    st.markdown('</div>', unsafe_allow_html=True)

def create_age_distribution(data, title, emoji):
    """Create compact age distribution"""
    if not data:
        return
    
    df = pd.DataFrame({
        'Age Group': [str(key).replace('dob_', '').upper() for key in data.keys()],
        'Cases': list(data.values())
    }).sort_values('Cases', ascending=False)
    
    st.markdown(f'<div class="chart-container">', unsafe_allow_html=True)
    st.markdown(f'<h4 style="color: #00d4ff; text-align: center;">{emoji} {title}</h4>', unsafe_allow_html=True)
    
    # Compact visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.bar_chart(df.set_index('Age Group')['Cases'], height=250)
    
    with col2:
        total = sum(data.values())
        for _, row in df.iterrows():
            percentage = (row['Cases'] / total) * 100
            st.metric(
                label=row['Age Group'],
                value=f"{row['Cases']}",
                delta=f"{percentage:.1f}%"
            )
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_compact_anomaly_results(viz_data):
    """Display compact anomaly detection results"""
    
    # Job Categories
    if 'job_categories' in viz_data:
        create_compact_bar_chart(
            viz_data['job_categories'],
            "Top Job Categories in Fraud",
            "👔"
        )
    
    # Age Groups
    if 'age_groups' in viz_data:
        create_age_distribution(
            viz_data['age_groups'],
            "Age Groups in Fraud",
            "👥"
        )
    
    # Transaction Categories
    if 'transaction_categories' in viz_data:
        create_compact_bar_chart(
            viz_data['transaction_categories'],
            "Top Transaction Categories",
            "🛒"
        )
    
    # Amount Analysis
    if 'amount_analysis' in viz_data:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<h4 style="color: #00d4ff; text-align: center;">💰 Amount Analysis</h4>', unsafe_allow_html=True)
        
        amt_data = viz_data['amount_analysis']
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Normal Avg Amount",
                value=f"${amt_data.get('normal_avg', 0):.2f}"
            )
        
        with col2:
            st.metric(
                label="Fraud Avg Amount", 
                value=f"${amt_data.get('fraud_avg', 0):.2f}"
            )
        
        with col3:
            increase_pct = amt_data.get('increase_pct', 0)
            st.metric(
                label="Amount Hike",
                value=f"{increase_pct:.1f}%",
                delta_color="inverse" if increase_pct > 0 else "normal"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)

# =============================================
# MAIN APP
# =============================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🏦 TruLedger.AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Financial Fraud Detection</p>', unsafe_allow_html=True)
    
    # Dataset Selection
    st.markdown('<h2 class="section-header">📁 Upload Documents</h2>', unsafe_allow_html=True)
    
    dataset_options = {
        "TransactionLogs-1",
        "TransactionLogs-2", 
        "TransactionLogs-3"
    }
    
    selected_key = st.selectbox(
        "💡 Choose a dataset to analyze:",
        options=list(dataset_options),
        help="Select from pre-uploaded datasets"
    )
    
    # Run Pipeline Button
    if st.button("🚀 Run Fraud Detection", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        input_file = os.path.join("Uploaded_Datasets", "Raw", f"{selected_key}.csv")
        
        if not os.path.exists(input_file):
            st.error(f"❌ Dataset file not found: {input_file}")
            return
        
        # Processing steps
        steps = [
            "🔄 Processing raw data...",
            "🔍 Running fraud detection...", 
            "📊 Loading results...",
            "✅ Analysis complete!"
        ]
        
        for i, step in enumerate(steps):
            status_text.markdown(f'<div class="processing-step">{step}</div>', unsafe_allow_html=True)
            progress_bar.progress((i + 1) * 25)
            
            if i == 0:
                processed_file = process_transaction_data(input_file)
                if processed_file is None:
                    st.error("❌ Failed to process data")
                    return
            elif i == 1:
                success = run_fraud_detection(processed_file)
                if not success:
                    st.error("❌ Failed to run fraud detection")
                    return
            elif i == 2:
                fraud_df = load_fraud_data()
                viz_data = load_visualization_data()
                llm_explanations = load_llm_explanations()
                
                if not fraud_df.empty:
                    st.session_state.analysis_complete = True
                    st.session_state.fraud_df = fraud_df
                    st.session_state.viz_data = viz_data
                    st.session_state.llm_explanations = llm_explanations.get('explanations', []) if llm_explanations else []
    
    # Results Section
    if st.session_state.get('analysis_complete', False):
        # Model Performance
        st.markdown('<h2 class="section-header">🤖 ML Model Performance</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Precision", "90.0%")
        with col2:
            st.metric("Recall", "74.0%") 
        with col3:
            st.metric("F1-Score", "0.80")
        
        # Anomaly Results
        st.markdown('<h2 class="section-header">📊 Fraud Detection Results</h2>', unsafe_allow_html=True)
        
        fraud_df = st.session_state.get('fraud_df', pd.DataFrame())
        st.info(f"**Detected {len(fraud_df)} fraudulent transactions**")
        
        viz_data = st.session_state.get('viz_data', {})
        if viz_data:
            display_compact_anomaly_results(viz_data)
        
        # Fraud Transactions
        if not fraud_df.empty:
            st.markdown('<h4 style="color: #00d4ff; margin: 1rem 0;">🔍 Recent Fraud Transactions</h4>', unsafe_allow_html=True)
            st.dataframe(fraud_df.head(8), use_container_width=True, height=300)
        
        # LLM Explanations
        if st.session_state.get('llm_explanations'):
            st.markdown('<h2 class="section-header">🧠 AI Explanations</h2>', unsafe_allow_html=True)
            
            explanations = st.session_state.get('llm_explanations', [])
            
            for i, explanation in enumerate(explanations[:3]):  # Show only 3 for compactness
                transaction_id = explanation.get('transaction_id', i + 1)
                risk_factors = explanation.get('risk_factors', [])
                explanation_text = explanation.get('explanation', 'No explanation available')
                
                st.markdown(f"""
                <div class="llm-card">
                    <h5>🚨 Transaction #{transaction_id}</h5>
                    <p><strong>Risk Factors:</strong> {', '.join(risk_factors)}</p>
                    <p><strong>Explanation:</strong> {explanation_text}</p>
                </div>
                """, unsafe_allow_html=True)

    # Technology Stack
    st.markdown('<h2 class="section-header">🛠️ Technology Stack</h2>', unsafe_allow_html=True)
    
    tech_cols = st.columns(2)
    with tech_cols[0]:
        st.write("**Data & ML Stack:**")
        st.write("• Pandas & PySpark")
        st.write("• TensorFlow & Scikit-learn") 
        st.write("• XGBoost & Joblib")
        st.write("• Jupyter Notebooks")
    
    with tech_cols[1]:
        st.write("**AI & Deployment:**")
        st.write("• LangChain & Llama 3.1")
        st.write("• Groq API")
        st.write("• Streamlit")
        st.write("• NumPy")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #94a3b8; padding: 1rem;'>
        <p><strong>Developed by Yashwanth Krishna Devanaboina</strong></p>
        <p>AI/ML Engineer | CS Student | AWS Certified</p>
        <p>© 2025 TruLedger.AI</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()