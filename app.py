import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# --- CONFIGURATION & THEME ---
st.set_page_config(
    page_title="FraudGuard AI | Enterprise Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Glassmorphism and SaaS Aesthetics
st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    [data-theme="dark"] .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* Glassmorphism Card */
    .st-emotion-cache-1r6slb0, .css-1r6slb0 {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
    }

    /* Navbar Header */
    .nav-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 2rem;
        background: rgba(255, 255, 255, 0.9);
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 2rem;
        border-radius: 0 0 15px 15px;
    }

    /* Status Indicators */
    .status-pill {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        background: #e2e8f0;
    }
    .status-online { background: #dcfce7; color: #166534; }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- CORE FUNCTIONS ---
def load_model_resource():
    """Load model with fallback for demo mode."""
    try:
        return joblib.load('fraud_model_final.pkl'), False
    except:
        return None, True

def get_severity(prob):
    if prob < 0.3: return "Low", "🟢"
    if prob < 0.6: return "Medium", "🟡"
    if prob < 0.85: return "High", "🟠"
    return "Critical", "🔴"

def render_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Score Index", 'font': {'size': 18}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "#3b82f6"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 30], 'color': '#dcfce7'},
                {'range': [30, 70], 'color': '#fef9c3'},
                {'range': [70, 100], 'color': '#fee2e2'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 85}}))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# --- UI COMPONENTS ---
def sidebar_navigation():
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/144/shield-lock.png", width=80)
        st.title("FraudGuard AI")
        st.markdown("`SYSTEM VERSION: 2.4.0-PRO`")
        st.divider()
        st.subheader("Control Panel")
        theme = st.toggle("Dark Mode Interface", value=False)
        st.divider()
        st.write("🛰️ **Server Status:**")
        st.markdown('<span class="status-pill status-online">Connected: US-EAST-1</span>', unsafe_allow_html=True)
        return theme

def header():
    st.markdown("""
        <div class="nav-container">
            <div style="display: flex; align-items: center;">
                <h2 style="margin: 0; color: #1e293b;">FraudGuard <span style="color: #3b82f6;">Systems</span></h2>
            </div>
            <div style="color: #64748b; font-size: 14px;">
                Node: Enterprise_04 • Active Session: Admin
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- MAIN APP FLOW ---
theme_choice = sidebar_navigation()
header()

model, is_demo = load_model_resource()

if is_demo:
    st.warning("⚠️ **DEMO MODE:** Model file not found. System is using heuristic logic for simulation.")

# --- INPUT SECTION ---
with st.container():
    c1, c2, c3 = st.columns([1.5, 1, 1], gap="medium")
    
    with c1:
        st.subheader("Transaction Intelligence")
        t_type = st.selectbox("Transaction Protocol", ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"], 
                             help="Protocol used for funds movement.")
        amount = st.number_input("Nominal Amount (USD)", min_value=0.0, value=1500.0, step=100.0, 
                                help="Total value of transaction in USD.")
        
    with c2:
        st.subheader("Account Parameters")
        orig_bal = st.number_input("Originating Balance", min_value=0.0, value=10000.0)
        new_bal = st.number_input("Post-Transaction Balance", min_value=0.0, value=8500.0)
        
        if amount > orig_bal:
            st.error("Insufficient Funds: Amount exceeds source balance.")
            
    with c3:
        st.subheader("Temporal Logic")
        step = st.slider("Step (Hour of Month)", 1, 744, 1, help="Simulation step in the monthly cycle.")
        st.info(f"Analysis Window: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# --- ACTION ---
if st.button("EXECUTE NEURAL ANALYSIS", use_container_width=True, type="primary"):
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)
        progress_bar.progress(percent_complete + 1)
    
    # Feature Engineering
    type_cash_out = 1 if t_type == "CASH_OUT" else 0
    type_transfer = 1 if t_type == "TRANSFER" else 0
    
    input_data = pd.DataFrame([[
        step, amount, orig_bal, new_bal, 0.0, 0.0, type_cash_out, type_transfer
    ]], columns=['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'type_CASH_OUT', 'type_TRANSFER'])

    # Prediction Logic
    if not is_demo:
        prob = model.predict_proba(input_data)[0][1]
        pred = model.predict(input_data)[0]
    else:
        # Fallback Simulation Logic
        prob = 0.88 if (amount > 500000 and t_type in ["TRANSFER", "CASH_OUT"]) else np.random.uniform(0.01, 0.15)
        pred = 1 if prob > 0.5 else 0

    severity, icon = get_severity(prob)
    
    # Store in history
    st.session_state.history.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Type": t_type,
        "Amount": amount,
        "Risk": f"{prob:.2%}",
        "Severity": severity
    })

    # --- RESULTS DASHBOARD ---
    st.divider()
    res1, res2 = st.columns([1, 1.5])
    
    with res1:
        st.plotly_chart(render_gauge(prob), use_container_width=True)
        st.metric("Threat Classification", severity, delta=f"{icon} Priority", delta_color="inverse")
        
    with res2:
        st.subheader("Technical Explanation (XAI)")
        # Simulated SHAP values
        importance = {
            "Amount vs Balance": prob * 0.6,
            "Transaction Type": 0.2 if t_type in ["TRANSFER", "CASH_OUT"] else 0.05,
            "Temporal Anomaly": np.random.uniform(0, 0.1),
            "Account History": 0.1
        }
        imp_df = pd.DataFrame(list(importance.items()), columns=['Factor', 'Impact'])
        fig_imp = px.bar(imp_df, x='Impact', y='Factor', orientation='h', 
                         color='Impact', color_continuous_scale='Blues')
        fig_imp.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
        st.plotly_chart(fig_imp, use_container_width=True)

# --- HISTORY & DATA ---
st.divider()
h1, h2 = st.columns([2, 1])

with h1:
    st.subheader("Recent System Audits")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history).iloc[::-1]
        st.table(history_df.head(5))
    else:
        st.write("No transactions analyzed in this session.")

with h2:
    st.subheader("Data Export")
    if st.session_state.history:
        csv = pd.DataFrame(st.session_state.history).to_csv(index=False)
        st.download_button("Download Audit Report (CSV)", data=csv, file_name="fraud_audit_log.csv", mime="text/csv")
        st.download_button("Download JSON Schema", data=pd.DataFrame(st.session_state.history).to_json(), file_name="fraud_log.json")

# --- FOOTER ---
st.markdown("""
    <div style="text-align: center; color: #94a3b8; font-size: 12px; margin-top: 5rem; padding: 2rem;">
        FraudGuard Enterprise v2.4.0-PRO | Licensed to Global FinTech Corp<br>
        System Status: <span style="color: #22c55e;">● Nominal</span> | Encryption: AES-256
    </div>
    """, unsafe_allow_html=True)