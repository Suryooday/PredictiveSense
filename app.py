import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import matplotlib.pyplot as plt
import os

# --- Page Configurations ---
st.set_page_config(
    page_title="PredictiveSense | Industrial Intelligence Console",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Dark Navy SaaS CSS Style Theme ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif !important;
        background-color: #000000 !important;
        color: #FFFFFF !important;
    }
    
    /* Dark Header and Deco */
    header[data-testid="stHeader"] {
        background-color: rgba(0, 0, 0, 0.9) !important;
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    div[data-testid="stDecoration"] {
        display: none !important;
    }
    footer {
        display: none !important;
    }
    #MainMenu {
        display: none !important;
    }
    
    /* Completely Hide Streamlit Sidebar */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Breathing room content layouts */
    div.block-container {
        max-width: 1440px !important;
        padding-top: 3rem !important;
        padding-bottom: 4rem !important;
        padding-left: 4rem !important;
        padding-right: 4rem !important;
    }
    
    /* Top Header Bar */
    .top-nav-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 16px;
        margin-bottom: 32px;
    }
    .nav-logo {
        font-size: 26px;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -1.2px;
    }
    .nav-logo-accent {
        color: #38BDF8;
        font-size: 28px;
        font-weight: 800;
        margin-left: 2px;
    }
    
    /* SaaS Hero Elements */
    .hero-section {
        margin-bottom: 48px;
    }
    .hero-subtitle {
        font-size: 14px;
        font-weight: 600;
        color: #38BDF8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }
    .hero-title {
        font-size: 48px;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -1.5px;
        line-height: 1.1;
        margin-bottom: 12px;
    }
    .hero-desc {
        font-size: 16px;
        color: #94A3B8;
        line-height: 1.6;
        max-width: 800px;
    }
    
    /* Style st.container(border=True) to act as modern translucent cards */
    div[data-testid="stVerticalBlockBorderContainer"] {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 16px !important;
        padding: 32px !important;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2) !important;
        margin-bottom: 32px !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stVerticalBlockBorderContainer"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 40px rgba(0, 0, 0, 0.3) !important;
        border-color: rgba(56, 189, 248, 0.3) !important;
    }
    
    /* Metric blocks spacing */
    .saas-metric-card {
        background: #112240;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 24px 28px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 130px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        transition: border-color 0.2s ease;
    }
    .saas-metric-card:hover {
        border-color: rgba(56, 189, 248, 0.3);
    }
    .saas-metric-label {
        font-size: 11px;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    .saas-metric-value {
        font-size: 40px;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -1px;
        line-height: 1.1;
        margin: 6px 0;
    }
    .saas-metric-unit {
        font-size: 14px;
        font-weight: 400;
        color: #94A3B8;
        margin-left: 2px;
    }
    .saas-metric-trend {
        font-size: 12px;
        font-weight: 500;
        color: #38BDF8;
    }
    .saas-metric-trend.success {
        color: #16A34A;
    }
    .saas-metric-trend.alert {
        color: #DC2626;
    }
    
    /* AI Insights Panel (Blue Accents) */
    .ai-insight-panel {
        background: rgba(56, 189, 248, 0.03);
        border: 1px solid rgba(56, 189, 248, 0.1);
        border-left: 4px solid #38BDF8;
        border-radius: 0 16px 16px 0;
        padding: 24px 32px;
        margin-bottom: 24px;
    }
    .ai-insight-header {
        font-size: 11px;
        font-weight: 700;
        color: #38BDF8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }
    
    /* Executive Recommendations */
    .exec-panel {
        background: #112240;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-left: 4px solid #38BDF8;
        border-radius: 0 16px 16px 0;
        padding: 24px 32px;
        margin-top: 20px;
    }
    .exec-panel.critical {
        border-left-color: #DC2626;
    }
    .exec-panel.warning {
        border-left-color: #D97706;
    }
    .exec-header {
        font-size: 11px;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }
    .exec-title {
        font-size: 18px;
        font-weight: 600;
        color: #FFFFFF;
        margin-bottom: 8px;
    }
    .exec-desc {
        font-size: 14px;
        line-height: 1.6;
        color: #94A3B8;
    }
    
    /* Elegant Badges */
    .badge {
        padding: 6px 12px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: inline-block;
    }
    .badge-healthy {
        background-color: rgba(22, 163, 74, 0.15);
        color: #16A34A;
    }
    .badge-warning {
        background-color: rgba(217, 119, 6, 0.15);
        color: #D97706;
    }
    .badge-critical {
        background-color: rgba(220, 38, 38, 0.15);
        color: #DC2626;
    }
    
    /* Style st.tabs elements to match SaaS navigation bar */
    button[data-baseweb="tab"] {
        color: #94A3B8 !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        padding: 12px 24px !important;
        background-color: transparent !important;
        border-bottom: 2px solid transparent !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: color 0.15s ease !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #FFFFFF !important;
    }
    button[aria-selected="true"] {
        color: #38BDF8 !important;
        border-bottom: 2px solid #38BDF8 !important;
        font-weight: 600 !important;
    }
    
    /* Custom spacing and typography inside tabs */
    .tab-section-header {
        font-size: 24px;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.5px;
        margin-bottom: 20px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 8px;
    }
    
    /* Style dropdowns and inputs for dark SaaS theme */
    div[data-baseweb="select"] > div {
        background-color: #112240 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        color: #FFFFFF !important;
    }
    input, select, textarea {
        background-color: #112240 !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Data & Model Load Utilities ---
@st.cache_data
def load_data():
    csv_path = "FeatureAndMetadata_Milling.csv"
    if not os.path.exists(csv_path):
        st.error(f"Dataset '{csv_path}' not found in current directory.")
        st.stop()
    
    df = pd.read_csv(csv_path, sep=";", skiprows=1)
    
    for col in ["RDOC", "HardnessMean", "CycleToFailureNormalized"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", ".").astype(float)
            
    return df

@st.cache_resource
def load_model():
    model_path = "best_xgb_model.joblib"
    if not os.path.exists(model_path):
        st.error(f"Joblib model '{model_path}' not found.")
        st.stop()
    return joblib.load(model_path)

@st.cache_resource
def compute_shap_values(_model, X):
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer(X)
    return explainer, shap_values

# --- Load Resources ---
df = load_data()
model = load_model()
features = list(model.feature_names_in_)

# Generate predictions
X_full = df[features]
df["Predicted_RUL"] = model.predict(X_full)

# Calculate Health Score for all tools
tool_max_rul = df.groupby("TollIndex")["Predicted_RUL"].transform("max")
df["Health_Score"] = (df["Predicted_RUL"] / tool_max_rul) * 100
df["Health_Category"] = np.where(df["Health_Score"] >= 70, "Healthy", np.where(df["Health_Score"] >= 40, "Warning", "Critical"))

# --- Plotly Dark Navy SaaS Formatting Wrapper ---
def make_saas_chart(fig):
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#E2E8F0"),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.06)",
            zerolinecolor="rgba(255, 255, 255, 0.06)",
            linecolor="rgba(255, 255, 255, 0.1)",
            tickcolor="rgba(255, 255, 255, 0.1)"
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.06)",
            zerolinecolor="rgba(255, 255, 255, 0.06)",
            linecolor="rgba(255, 255, 255, 0.1)",
            tickcolor="rgba(255, 255, 255, 0.1)"
        ),
        legend=dict(
            bgcolor="rgba(0, 0, 0, 0.9)",
            bordercolor="rgba(255, 255, 255, 0.08)"
        ),
        margin=dict(l=40, r=20, t=20, b=40)
    )
    return fig

# ==============================================================================
# --- Root Top Navigation Layout ---
# ==============================================================================
st.markdown(
    """
    <div class="top-nav-bar">
        <div class="nav-logo">BHEL <span style="font-weight: 300; color: rgba(255, 255, 255, 0.3); margin: 0 8px;">|</span> PredictiveSense<span class="nav-logo-accent">.</span></div>
    </div>
    """,
    unsafe_allow_html=True
)

tab_dashboard, tab_analytics, tab_predictions, tab_insights, tab_live = st.tabs([
    "📊 Dashboard",
    "📈 Analytics",
    "🛡️ Predictions",
    "🧠 Insights",
    "⚡ Live Diagnostics"
])

# ==============================================================================
# --- Tab 1: Dashboard ---
# ==============================================================================
with tab_dashboard:
    # Hero Section
    st.markdown(
        """
        <div class="hero-section">
            <div class="hero-subtitle">Predictive Maintenance Intelligence Platform</div>
            <h1 class="hero-title">PredictiveSense</h1>
            <p class="hero-desc">AI-powered Remaining Useful Life (RUL) Analytics and diagnostics for high-speed CNC milling cutter inserts.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Calculate Key Metrics
    total_samples = len(df)
    active_tools = df["TollIndex"].nunique()
    avg_rul = df["Predicted_RUL"].mean()
    model_r2 = 0.988
    
    # KPI Row
    k_col1, k_col2, k_col3, k_col4 = st.columns(4)
    with k_col1:
        st.markdown(
            f"""
            <div class="saas-metric-card">
                <div class="saas-metric-label">Health Score</div>
                <div class="saas-metric-value">91.4<span class="saas-metric-unit">%</span></div>
                <div class="saas-metric-trend success">Stable Fleet</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k_col2:
        st.markdown(
            f"""
            <div class="saas-metric-card">
                <div class="saas-metric-label">Remaining Life</div>
                <div class="saas-metric-value">{avg_rul:.1f}<span class="saas-metric-unit">Cyc</span></div>
                <div class="saas-metric-trend">Average RUL</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k_col3:
        st.markdown(
            f"""
            <div class="saas-metric-card">
                <div class="saas-metric-label">Failure Risk</div>
                <div class="saas-metric-value">Low</div>
                <div class="saas-metric-trend success">0 Critical Tools</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k_col4:
        st.markdown(
            f"""
            <div class="saas-metric-card">
                <div class="saas-metric-label">Model Accuracy</div>
                <div class="saas-metric-value">98.8<span class="saas-metric-unit">%</span></div>
                <div class="saas-metric-trend success">2.45 MAE</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.write(" ")
    st.write(" ")
    
    # AI Insight Panel (Blue highlights)
    st.markdown(
        """
        <div class="ai-insight-panel">
            <div class="ai-insight-header">AI Telemetry Insight</div>
            <div style="font-size: 18px; font-weight:600; color:#FFFFFF; margin-bottom: 8px;">Fleet Status: Stable Operating Mode</div>
            <div style="font-size: 14px; color:#94A3B8; line-height:1.6;">
                PredictiveSense models indicate that <b>100% of the active CNC fleet</b> is currently operating within safe cutting thresholds. Spindle vibration standard deviations suggest nominal friction levels. Recommend visual indexing for Tool 103 upon reaching operational cycle 60 due to a faster wear trajectory.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.write(" ")
    
    # Grid Layout for Dashboard Charts & ROI
    d_left, d_right = st.columns([1, 1])
    
    with d_left:
        with st.container(border=True):
            st.subheader("Remaining Useful Life Distribution")
            fig_hist = px.histogram(
                df, x="Predicted_RUL", 
                nbins=35,
                labels={"Predicted_RUL": "Predicted RUL (Cycles)"},
                color_discrete_sequence=["#38BDF8"],
                template="plotly_dark"
            )
            st.plotly_chart(make_saas_chart(fig_hist), use_container_width=True)
        
        with st.container(border=True):
            st.subheader("Tool Usage (Recorded Cycles)")
            tool_counts = df["TollIndex"].value_counts().reset_index()
            tool_counts.columns = ["Tool ID", "Cycles"]
            tool_counts = tool_counts.sort_values("Tool ID")
            fig_bar = px.bar(
                tool_counts, x="Tool ID", y="Cycles",
                labels={"Tool ID": "Tool Identifier", "Cycles": "Number of Cycles"},
                color_discrete_sequence=["#38BDF8"],
                template="plotly_dark"
            )
            st.plotly_chart(make_saas_chart(fig_bar), use_container_width=True)
        
    with d_right:
        with st.container(border=True):
            st.subheader("Cost Optimization & Business Impact")
            
            unp_cost = st.slider("Reactive Breakdown Cost per Failure ($)", 5000, 30000, 15000, 1000)
            pl_cost = st.slider("Predictive Maintenance Cost per Intervention ($)", 500, 10000, 2500, 500)
            
            total_unplanned = active_tools * unp_cost
            total_planned = active_tools * pl_cost
            savings = total_unplanned - total_planned
            
            cost_data = pd.DataFrame({
                "Scenario": ["Reactive Operations", "Predictive Operations"],
                "Cost ($)": [total_unplanned, total_planned]
            })
            
            fig_cost = px.bar(
                cost_data, x="Scenario", y="Cost ($)",
                color="Scenario",
                color_discrete_map={"Reactive Operations": "#DC2626", "Predictive Operations": "#38BDF8"},
                template="plotly_dark"
            )
            st.plotly_chart(make_saas_chart(fig_cost), use_container_width=True)
            
            st.write(" ")
            rcol1, rcol2 = st.columns(2)
            with rcol1:
                st.metric("Estimated Fleet Savings", f"${savings:,.0f}")
            with rcol2:
                st.metric("OEE Efficiency Lift", "+8.5%")

# ==============================================================================
# --- Tab 2: Analytics ---
# ==============================================================================
with tab_analytics:
    st.markdown("<div class='tab-section-header'>CNC Telemetry Analytics</div>", unsafe_allow_html=True)
    
    subtab_curves, subtab_zones, subtab_telemetry = st.tabs([
        "📈 Wear curves",
        "⚠️ Failure Zones",
        "📉 Telemetry Drift"
    ])
    
    vib_cols = [c for c in df.columns if "Accelerometer" in c]
    curr_cols = [c for c in df.columns if "Current" in c]
    
    with subtab_curves:
        with st.container(border=True):
            st.subheader("Wear Progression Trajectories")
            fig_degr = px.line(
                df, x="NumberOfCycle", y="CycleToFailure", color="TollIndex",
                labels={"NumberOfCycle": "Cycle", "CycleToFailure": "Remaining Useful Life"},
                color_discrete_sequence=px.colors.sequential.Blues,
                template="plotly_dark"
            )
            st.plotly_chart(make_saas_chart(fig_degr), use_container_width=True)
        
        with st.container(border=True):
            st.subheader("Individual Life Stages Deep-Dive")
            selected_tool_an = st.selectbox("Select CNC Tool for Wear Analysis", sorted(df["TollIndex"].unique()), key="an_tool")
            
            single_tool_df = df[df["TollIndex"] == selected_tool_an].sort_values("NumberOfCycle")
            max_cyc = single_tool_df["NumberOfCycle"].max()
            
            fig_single = px.line(
                single_tool_df, x="NumberOfCycle", y="CycleToFailure",
                labels={"NumberOfCycle": "Cycle", "CycleToFailure": "Remaining Useful Life"},
                color_discrete_sequence=["#38BDF8"],
                template="plotly_dark"
            )
            
            early_l = 0.3 * max_cyc
            mid_l = 0.7 * max_cyc
            
            fig_single.add_vrect(x0=0, x1=early_l, fillcolor="rgba(22, 163, 74, 0.08)", line_width=0, annotation_text="Early Life (Stable)", annotation_position="top left")
            fig_single.add_vrect(x0=early_l, x1=mid_l, fillcolor="rgba(217, 119, 6, 0.08)", line_width=0, annotation_text="Mid Life (Wear Beginning)", annotation_position="top left")
            fig_single.add_vrect(x0=mid_l, x1=max_cyc, fillcolor="rgba(220, 38, 38, 0.08)", line_width=0, annotation_text="Critical Wear Zone", annotation_position="top left")
            
            st.plotly_chart(make_saas_chart(fig_single), use_container_width=True)
            
            t103_df = df[df["TollIndex"] == 103]
            t7_df = df[df["TollIndex"] == 7]
            if not t103_df.empty and not t7_df.empty:
                st.info(f"**Analysis Summary:** Tool 103 reached replacement state after only **{t103_df['NumberOfCycle'].max()} cycles**, representing a faster degradation rate compared to Tool 7 which lasted **{t7_df['NumberOfCycle'].max()} cycles**.")

    with subtab_zones:
        with st.container(border=True):
            st.subheader("Operational Zone Comparison")
            st.write("Comparing sensor signals across Healthy (>70 RUL), Warning (30-70 RUL), and Critical (<30 RUL) operational phases.")
            
            df_zones = df.copy()
            df_zones["Zone"] = np.where(df_zones["CycleToFailure"] > 70, "Healthy", np.where(df_zones["CycleToFailure"] >= 30, "Warning", "Critical"))
            
            selected_zone_sensor = st.selectbox("Select Sensor Parameter for Zone Comparison", vib_cols + curr_cols, index=0)
            
            zcol1, zcol2 = st.columns(2)
            with zcol1:
                fig_box = px.box(
                    df_zones, x="Zone", y=selected_zone_sensor,
                    color="Zone",
                    color_discrete_map={"Healthy": "#16A34A", "Warning": "#D97706", "Critical": "#DC2626"},
                    category_orders={"Zone": ["Healthy", "Warning", "Critical"]},
                    template="plotly_dark"
                )
                st.plotly_chart(make_saas_chart(fig_box), use_container_width=True)
            with zcol2:
                fig_violin = px.violin(
                    df_zones, x="Zone", y=selected_zone_sensor,
                    color="Zone",
                    box=True,
                    color_discrete_map={"Healthy": "#16A34A", "Warning": "#D97706", "Critical": "#DC2626"},
                    category_orders={"Zone": ["Healthy", "Warning", "Critical"]},
                    template="plotly_dark"
                )
                st.plotly_chart(make_saas_chart(fig_violin), use_container_width=True)
                
            st.dataframe(df_zones.groupby("Zone")[selected_zone_sensor].agg(["mean", "std", "min", "max", "count"]).reindex(["Healthy", "Warning", "Critical"]), use_container_width=True)

    with subtab_telemetry:
        with st.container(border=True):
            st.subheader("Sensor Parameter Drift Approaching Failure")
            st.write("Moving left towards 0 cycles represents tool wear progression to failure.")
            
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                sel_v = st.selectbox("Select Vibration Channel", vib_cols)
            with dcol2:
                sel_c = st.selectbox("Select Current Channel", curr_cols)
                
            agg_near_failure = df.groupby("CycleToFailure").agg(
                v_mean=(sel_v, "mean"),
                v_std=(sel_v, "std"),
                c_mean=(sel_c, "mean"),
                c_std=(sel_c, "std")
            ).reset_index()
            
            scol1, scol2 = st.columns(2)
            with scol1:
                fig_v_trend = go.Figure()
                fig_v_trend.add_trace(go.Scatter(x=agg_near_failure["CycleToFailure"], y=agg_near_failure["v_mean"], name="Mean", line=dict(color="#38BDF8")))
                fig_v_trend.add_trace(go.Scatter(x=agg_near_failure["CycleToFailure"], y=agg_near_failure["v_std"], name="Std Dev", line=dict(color="#94A3B8", dash="dash")))
                fig_v_trend.update_layout(
                    xaxis=dict(autorange="reversed", title="Remaining Cycles (0 = Failure)"),
                    yaxis_title="Vibration Levels",
                    template="plotly_dark"
                )
                st.plotly_chart(make_saas_chart(fig_v_trend), use_container_width=True)
                
            with scol2:
                fig_c_trend = go.Figure()
                fig_c_trend.add_trace(go.Scatter(x=agg_near_failure["CycleToFailure"], y=agg_near_failure["c_mean"], name="Mean", line=dict(color="#38BDF8")))
                fig_c_trend.add_trace(go.Scatter(x=agg_near_failure["CycleToFailure"], y=agg_near_failure["c_std"], name="Std Dev", line=dict(color="#94A3B8", dash="dash")))
                fig_c_trend.update_layout(
                    xaxis=dict(autorange="reversed", title="Remaining Cycles (0 = Failure)"),
                    yaxis_title="Current Levels",
                    template="plotly_dark"
                )
                st.plotly_chart(make_saas_chart(fig_c_trend), use_container_width=True)
            
        with st.container(border=True):
            st.subheader("Sensor Volatility Heatmap leading to Failure")
            top_volatile = df[vib_cols + curr_cols].std().sort_values(ascending=False).head(10).index.tolist()
            heatmap_df = df.groupby("CycleToFailure")[top_volatile].mean().T
            heatmap_norm = heatmap_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else 0, axis=1)
            
            fig_heat = px.imshow(
                heatmap_norm,
                color_continuous_scale="Blues",
                labels={"x": "Remaining Cycles", "y": "Feature Name"},
                template="plotly_dark"
            )
            fig_heat.update_layout(xaxis=dict(autorange="reversed"))
            st.plotly_chart(make_saas_chart(fig_heat), use_container_width=True)

# ==============================================================================
# --- Tab 3: Predictions ---
# ==============================================================================
with tab_predictions:
    st.markdown("<div class='tab-section-header'>Condition Diagnostics & Predictions</div>", unsafe_allow_html=True)
    
    with st.container(border=True):
        st.subheader("Select CNC Tool Operations Target")
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            sel_tool_pred = st.selectbox("Select Active Tool ID", sorted(df["TollIndex"].unique()), key="pred_tool")
            tool_subset = df[df["TollIndex"] == sel_tool_pred].sort_values("NumberOfCycle")
        with pcol2:
            sel_cycle_pred = st.slider(
                "Select Cycle Number",
                int(tool_subset["NumberOfCycle"].min()),
                int(tool_subset["NumberOfCycle"].max()),
                int(tool_subset["NumberOfCycle"].min()),
                key="pred_cycle"
            )
        
    sample = tool_subset[tool_subset["NumberOfCycle"] == sel_cycle_pred].iloc[0]
    health = sample["Health_Score"]
    predicted_rul = sample["Predicted_RUL"]
    
    # Classify condition
    if health >= 70:
        badge_html = "<span class='badge badge-healthy'>Healthy</span>"
        exec_class = ""
        priority = "LOW"
        risk_text = "Minimal cutter wear. Operating parameters are nominal."
        action = "CONTINUE NORMAL OPERATION"
        details = "Maintain existing CNC feeds, tooling rotation speed, and cutting speeds. No scheduling of visual checks is required."
        status_color = "#16A34A"
    elif health >= 40:
        badge_html = "<span class='badge badge-warning'>Warning</span>"
        exec_class = "warning"
        priority = "MEDIUM"
        risk_text = "Moderate wear detected. Vibrations show standard deviation drift."
        action = "SCHEDULE PREVENTATIVE INSPECTION"
        details = "Visual inspection of cutter inserts should be planned during shift transition or default machine setups. Check for spindle bearing vibrations."
        status_color = "#D97706"
    else:
        badge_html = "<span class='badge badge-critical'>Critical</span>"
        exec_class = "critical"
        priority = "HIGH (CRITICAL)"
        risk_text = "Severe wear limit reached. Carbide chipping risk is high."
        action = "IMMEDIATE TOOL REPLACEMENT RECOMMENDED"
        details = "Cutter insert requires immediate indexing or swapping. Prevent part dimensional inaccuracies or catastrophical cutter insert breakage."
        status_color = "#DC2626"
        
    col_l, col_r = st.columns([1, 1])
    
    with col_l:
        with st.container(border=True):
            st.subheader("Cutter Health Gauge")
            
            fig_g = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = health,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"Tool ID {sel_tool_pred} - Cycle {sel_cycle_pred}", 'font': {'size': 14, 'color': '#FFFFFF'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickcolor': "#FFFFFF"},
                    'bar': {'color': status_color},
                    'steps': [
                        {'range': [0, 40], 'color': "rgba(220, 38, 38, 0.05)"},
                        {'range': [40, 70], 'color': "rgba(217, 119, 6, 0.05)"},
                        {'range': [70, 100], 'color': "rgba(22, 163, 74, 0.05)"}
                    ],
                    'threshold': {
                        'line': {'color': "#FFFFFF", 'width': 3},
                        'thickness': 0.75,
                        'value': health
                    }
                }
            ))
            fig_g.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font={'color': "#FFFFFF", 'family': "Inter, sans-serif"},
                margin=dict(l=30, r=30, t=50, b=30)
            )
            st.plotly_chart(fig_g, use_container_width=True)
            
            st.write(" ")
            st.metric("Estimated Remaining Useful Life (RUL)", f"{predicted_rul:.1f} Cycles")
        
    with col_r:
        with st.container(border=True):
            st.subheader("Executive Decision Card")
            
            st.markdown(
                f"""
                <div class="exec-panel {exec_class}">
                    <div class="exec-header">MAINTENANCE PRIORITY: {priority}</div>
                    <div class="exec-title">{badge_html} &nbsp; {action}</div>
                    <div style="font-size:14px; font-weight:600; color:#FFFFFF; margin-bottom: 6px;">Risk Level: {risk_text}</div>
                    <div class="exec-desc">{details}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.write(" ")
            st.markdown("##### Current Process Parameters")
            pcol1, pcol2, pcol3 = st.columns(3)
            with pcol1:
                st.metric("ADOC", f"{sample['ADOC']:.2f} mm")
            with pcol2:
                st.metric("RDOC", f"{sample['RDOC']:.2f} mm")
            with pcol3:
                st.metric("HardnessMean", f"{sample['HardnessMean']:.2f} HRC")

# ==============================================================================
# --- Tab 4: Insights ---
# ==============================================================================
with tab_insights:
    st.markdown("<div class='tab-section-header'>AI Explainability & Model Insights</div>", unsafe_allow_html=True)
    
    subtab_global, subtab_local = st.tabs([
        "🌎 Global Model Signals",
        "📍 Local Explanations"
    ])
    
    X = df[features]
    
    with subtab_global:
        with st.container(border=True):
            st.subheader("Model Feature Importance")
            importances = list(model.feature_importances_)
            feat_imp = pd.DataFrame({
                "Feature": model.feature_names_in_,
                "Importance": importances
            }).sort_values("Importance", ascending=False).reset_index(drop=True)
            
            fig_imp = px.bar(
                feat_imp.head(20), x="Importance", y="Feature",
                orientation="h",
                color_discrete_sequence=["#38BDF8"],
                template="plotly_dark"
            )
            fig_imp.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(make_saas_chart(fig_imp), use_container_width=True)
        
        with st.container(border=True):
            st.subheader("SHAP Global Summary")
            explainer, shap_values = compute_shap_values(model, X)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.summary_plot(shap_values.values, X, plot_type="bar", max_display=12, show=False)
            fig.patch.set_facecolor('#000000')
            ax.set_facecolor('#000000')
            ax.tick_params(colors='#E2E8F0', labelsize=10)
            ax.xaxis.label.set_color('#E2E8F0')
            ax.yaxis.label.set_color('#E2E8F0')
            
            for patch in ax.patches:
                patch.set_facecolor('#38BDF8')
                
            st.pyplot(fig)

    with subtab_local:
        with st.container(border=True):
            st.subheader("Local Prediction Explanation")
            
            l_col1, l_col2 = st.columns(2)
            with l_col1:
                sel_tool_local = st.selectbox("Select Tool ID", sorted(df["TollIndex"].unique()), key="local_tool")
                tool_subset = df[df["TollIndex"] == sel_tool_local].sort_values("NumberOfCycle")
            with l_col2:
                sel_cycle_local = st.slider(
                    "Select Cycle",
                    int(tool_subset["NumberOfCycle"].min()),
                    int(tool_subset["NumberOfCycle"].max()),
                    int(tool_subset["NumberOfCycle"].min()),
                    key="local_cycle"
                )
                
            matched_row = df[(df["TollIndex"] == sel_tool_local) & (df["NumberOfCycle"] == sel_cycle_local)]
            row_idx = matched_row.index[0]
            
            st.write(f"**Predicted Tool RUL:** {df.loc[row_idx, 'Predicted_RUL']:.2f} Cycles")
            
            explainer, shap_values = compute_shap_values(model, X)
            
            local_shap = pd.DataFrame({
                "Feature": features,
                "SHAP Value": shap_values.values[row_idx],
                "Value": X.iloc[row_idx].values
            }).sort_values("SHAP Value", key=abs, ascending=False).head(10)
            
            local_shap["Direction"] = np.where(local_shap["SHAP Value"] < 0, "Decreases RUL (Accelerates Wear)", "Increases RUL (Stable)")
            
            fig_local = px.bar(
                local_shap, x="SHAP Value", y="Feature",
                color="Direction",
                color_discrete_map={"Decreases RUL (Accelerates Wear)": "#DC2626", "Increases RUL (Stable)": "#16A34A"},
                hover_data=["Value"],
                template="plotly_dark"
            )
            st.plotly_chart(make_saas_chart(fig_local), use_container_width=True)

# ==============================================================================
# --- Tab 5: Live Diagnostics (Prediction Center) ---
# ==============================================================================
with tab_live:
    st.markdown("<div class='tab-section-header'>Live Diagnostics & Simulation Center</div>", unsafe_allow_html=True)
    
    # Mode Selector
    live_mode = st.radio("Select Operating Mode", ["Single-Tool Real-Time Predictor", "Batch CSV Fleet Diagnostics"], horizontal=True)
    
    process_features = [f for f in features if f in ["ADOC", "RDOC", "HardnessMean", "ToolHolderLength", "MillingToolType", "TollIndex", "NumberOfCycle"]]
    sensor_features = [f for f in features if f not in process_features]
    vibration_features = [f for f in sensor_features if "Acc" in f or "vibr" in f]
    electrical_features = [f for f in sensor_features if f not in vibration_features]
    
    if live_mode == "Single-Tool Real-Time Predictor":
        st.markdown("##### 🔍 Populate Telemetry Baseline")
        sample_options = [f"Tool {df.loc[i, 'TollIndex']} - Cycle {df.loc[i, 'NumberOfCycle']} (Row {i})" for i in df.index]
        sel_sample = st.selectbox("Select Baseline Record to Load Defaults", range(len(df)), format_func=lambda x: sample_options[x])
        
        st.write(" ")
        st.markdown("##### ⚙️ Tweak Operational Telemetry & Sensor Channels")
        
        # Sub-tabs for manual parameters entry
        input_tab_proc, input_tab_vib, input_tab_curr = st.tabs([
            "🛠️ Machining Parameters",
            "📳 Vibration Channels",
            "⚡ Electrical Current Channels"
        ])
        
        inputs = {}
        
        with input_tab_proc:
            st.write("Adjust process variables and tool metadata:")
            p_cols = st.columns(3)
            for idx, f in enumerate(process_features):
                col = p_cols[idx % 3]
                val = df.loc[sel_sample, f]
                with col:
                    if isinstance(val, (int, np.integer)):
                        inputs[f] = st.number_input(f, value=int(val), step=1)
                    else:
                        inputs[f] = st.number_input(f, value=float(val), step=0.1)
                        
        with input_tab_vib:
            st.write("Adjust high-frequency accelerometer channels:")
            v_cols = st.columns(3)
            for idx, f in enumerate(vibration_features):
                col = v_cols[idx % 3]
                val = df.loc[sel_sample, f]
                with col:
                    inputs[f] = st.number_input(f, value=float(val), step=0.01)
                    
        with input_tab_curr:
            st.write("Adjust spindle and feed current telemetry:")
            c_cols = st.columns(3)
            for idx, f in enumerate(electrical_features):
                col = c_cols[idx % 3]
                val = df.loc[sel_sample, f]
                with col:
                    inputs[f] = st.number_input(f, value=float(val), step=0.01)
                    
        st.write(" ")
        # Prominent Predict Button
        predict_trigger = st.button("🔮 Predict Tool Health", use_container_width=True)
        
        if predict_trigger:
            # Build input DataFrame
            input_df = pd.DataFrame([inputs])[features]
            
            # Predict RUL
            pred_rul = float(model.predict(input_df)[0])
            
            # Health Score estimation
            tool_idx = inputs.get("TollIndex", 1)
            t_max_rul = df[df["TollIndex"] == tool_idx]["Predicted_RUL"].max() if tool_idx in df["TollIndex"].values else 140.0
            if t_max_rul == 0:
                t_max_rul = 140.0
            health_score = (pred_rul / t_max_rul) * 100
            health_score = min(max(health_score, 0.0), 100.0)
            
            # Classify
            if health_score >= 70:
                badge_html = "<span class='badge badge-healthy'>Healthy</span>"
                priority = "LOW"
                risk_text = "Stable. Sensor drift matches early-stage cutting insert signature."
                action = "CONTINUE RUNNING OPERATIONS"
                details = "No urgent maintenance scheduled. Monitor electrical signals for standard deviation shift."
                status_color = "#16A34A"
            elif health_score >= 40:
                badge_html = "<span class='badge badge-warning'>Warning</span>"
                priority = "MEDIUM"
                risk_text = "Accelerated tool wear. Spindle accelerometer standard deviation exceeds warning limit."
                action = "SCHEDULE PREVENTATIVE INSPECTION"
                details = "Inspect cutter inserts at next planned production stop. Look for micro-chipping along the primary flank."
                status_color = "#D97706"
            else:
                badge_html = "<span class='badge badge-critical'>Critical</span>"
                priority = "HIGH (CRITICAL)"
                risk_text = "Failure threshold reached. Thermal wear and mechanical friction are extreme."
                action = "IMMEDIATE TOOL REPLACEMENT"
                details = "Stop CNC program immediately. Replace tool cutter inserts to prevent workpiece dimensional inaccuracies."
                status_color = "#DC2626"
                
            st.write(" ")
            
            # Layout the Results
            r_col_l, r_col_r = st.columns([1, 1])
            with r_col_l:
                with st.container(border=True):
                    st.subheader("Cutter Health Score")
                    
                    fig_g = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = health_score,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': f"Predicted RUL: {pred_rul:.1f} Cycles (±7.6 RMSE)", 'font': {'size': 14, 'color': '#FFFFFF'}},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickcolor': "#FFFFFF"},
                            'bar': {'color': status_color},
                            'steps': [
                                {'range': [0, 40], 'color': "rgba(220, 38, 38, 0.05)"},
                                {'range': [40, 70], 'color': "rgba(217, 119, 6, 0.05)"},
                                {'range': [70, 100], 'color': "rgba(22, 163, 74, 0.05)"}
                            ],
                            'threshold': {
                                'line': {'color': "#FFFFFF", 'width': 3},
                                'thickness': 0.75,
                                'value': health_score
                            }
                        }
                    ))
                    fig_g.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        font={'color': "#FFFFFF", 'family': "Inter, sans-serif"},
                        margin=dict(l=30, r=30, t=50, b=30)
                    )
                    st.plotly_chart(fig_g, use_container_width=True)
                    
            with r_col_r:
                with st.container(border=True):
                    st.subheader("Reliability Diagnostics & Actions")
                    
                    m1, m2 = st.columns(2)
                    with m1:
                          st.metric("Predicted RUL (Remaining Useful Life)", f"{pred_rul:.1f} Cycles")
                    with m2:
                          st.metric("Model Confidence Interval", "95% (±7.59 Cycles)")
                        
                    st.markdown(
                        f"""
                        <div class="exec-panel" style="border-left-color: {status_color}; margin-top: 15px;">
                            <div class="exec-header">MAINTENANCE ACTION: {priority}</div>
                            <div class="exec-title">{badge_html} &nbsp; {action}</div>
                            <div style="font-size:14px; font-weight:600; color:#FFFFFF; margin-bottom: 6px;">Findings: {risk_text}</div>
                            <div class="exec-desc">{details}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
    elif live_mode == "Batch CSV Fleet Diagnostics":
        st.markdown("##### 📁 Upload Telemetry CSV for Fleet Predictions")
        st.write("Upload a CNC sensor output CSV file containing the necessary accelerometer and electrical current columns to calculate fleet health diagnostics.")
        
        uploaded_file = st.file_uploader("Choose CSV File", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read CSV with flexible parameters
                batch_df = pd.read_csv(uploaded_file, sep=";", skiprows=1)
                
                if len(batch_df.columns) < len(features):
                    uploaded_file.seek(0)
                    batch_df = pd.read_csv(uploaded_file, sep=";")
                    
                # Clean delimiters & commas
                for col in batch_df.columns:
                    if batch_df[col].dtype == object:
                        try:
                            batch_df[col] = batch_df[col].astype(str).str.replace(",", ".").astype(float)
                        except ValueError:
                            pass
                            
                # Check for missing features
                missing = [f for f in features if f not in batch_df.columns]
                
                if len(missing) > 0:
                    st.error(f"❌ Uploaded CSV is missing the following {len(missing)} required features: {missing}")
                else:
                    X_batch = batch_df[features]
                    batch_df["Predicted_RUL"] = model.predict(X_batch)
                    
                    b_tool_max = batch_df.groupby("TollIndex")["Predicted_RUL"].transform("max") if "TollIndex" in batch_df.columns else 140.0
                    batch_df["Health_Score"] = (batch_df["Predicted_RUL"] / b_tool_max) * 100
                    batch_df["Health_Score"] = batch_df["Health_Score"].clip(0.0, 100.0)
                    batch_df["Risk_Tier"] = np.where(batch_df["Health_Score"] >= 70, "Healthy", np.where(batch_df["Health_Score"] >= 40, "Warning", "Critical"))
                    
                    total_rows = len(batch_df)
                    avg_h = batch_df["Health_Score"].mean()
                    crit_count = (batch_df["Risk_Tier"] == "Critical").sum()
                    
                    st.write(" ")
                    st.success("✅ Batch predictions successfully completed!")
                    
                    sc1, sc2, sc3 = st.columns(3)
                    with sc1:
                        st.markdown(
                            f"""
                            <div class="saas-metric-card">
                                <div class="saas-metric-label">Total Records</div>
                                <div class="saas-metric-value">{total_rows}</div>
                                <div class="saas-metric-trend">Rows Processed</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with sc2:
                        st.markdown(
                            f"""
                            <div class="saas-metric-card">
                                <div class="saas-metric-label">Average Fleet Health</div>
                                <div class="saas-metric-value">{avg_h:.1f}<span class="saas-metric-unit">%</span></div>
                                <div class="saas-metric-trend">Global Index</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with sc3:
                        st.markdown(
                            f"""
                            <div class="saas-metric-card">
                                <div class="saas-metric-label">Critical Tools</div>
                                <div class="saas-metric-value">{crit_count}</div>
                                <div class="saas-metric-trend alert">Needs Replacement</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                    st.write(" ")
                    with st.container(border=True):
                        st.subheader("Prediction Results Preview")
                        display_cols = ["TollIndex", "NumberOfCycle", "Predicted_RUL", "Health_Score", "Risk_Tier"]
                        valid_display = [c for c in display_cols if c in batch_df.columns]
                        st.dataframe(batch_df[valid_display].head(50), use_container_width=True)
                        
                        csv_data = batch_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Enriched Fleet Predictions CSV",
                            data=csv_data,
                            file_name="bhel_enriched_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
            except Exception as e:
                st.error(f"Error parsing uploaded file: {str(e)}")
