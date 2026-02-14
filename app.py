import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
import shap

st.set_page_config(
    page_title="⚡ Intelligent Abnormal Electricity Usage Detection",
    layout="wide"
)


st.markdown("""
<style>
h1 { font-size: 42px !important; }
h2 { font-size: 32px !important; }
h3 { font-size: 26px !important; }
p, li { font-size: 20px !important; }
.center { text-align: center; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    model = joblib.load("abnormal_usage_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = load_explainer(model)


st.markdown(
    """
    <h1 class="center">⚡ Intelligent Abnormal Electricity Usage Detection</h1>
    <p class="center">
    Machine Learning system for detecting abnormal residential electricity consumption
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()


st.subheader("🏠 Household Input Parameters")

c1, c2, c3 = st.columns(3)

with c1:
    region = st.selectbox("📍 Region", ["IN_KL_ALP", "IN_KL_ERN", "IN_KL_TVM"])
    dwelling = st.selectbox("🏢 Dwelling Type", ["Apartment", "Independent House", "Villa"])
    occupants = st.number_input("👨‍👩‍👧 Number of Occupants", 1, 10, 4)

with c2:
    house_area = st.number_input("🏠 House Area (sqft)", 300, 5000, 1200)
    appliance_score = st.number_input("🔌 Appliance Score", 1, 30, 12)
    connected_load = st.number_input("⚡ Connected Load (kW)", 1.0, 20.0, 5.0)

with c3:
    temperature = st.number_input("🌡 Temperature (°C)", 10.0, 50.0, 30.0)
    humidity = st.number_input("💧 Humidity (%)", 20.0, 100.0, 65.0)
    deviation_abs = st.number_input("📉 Usage Deviation (kWh)", 0.0, 50.0, 5.0)


expected_energy = max(1.0, connected_load * 5)
actual_energy = expected_energy + deviation_abs

usage_ratio = actual_energy / expected_energy
load_utilization = actual_energy / max(connected_load, 0.1)

region_map = {"IN_KL_ALP": 0, "IN_KL_ERN": 1, "IN_KL_TVM": 2}
dwelling_map = {"Apartment": 0, "Independent House": 1, "Villa": 2}


st.divider()
st.markdown("<div class='center'>", unsafe_allow_html=True)
predict_btn = st.button("🔍 Analyze Usage Risk")
st.markdown("</div>", unsafe_allow_html=True)


if predict_btn:

    input_data = np.array([[
        region_map[region],
        dwelling_map[dwelling],
        occupants,
        house_area,
        appliance_score,
        connected_load,
        temperature,
        humidity,
        deviation_abs,
        usage_ratio,
        load_utilization
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    normal_prob = proba[0] * 100
    abnormal_prob = proba[1] * 100

   
    st.divider()
    st.subheader("📄 Prediction Result")

    if prediction == 1:
        st.error("⚠️ Abnormal Electricity Usage Detected")
    else:
        st.success("✅ Normal Electricity Usage Detected")

    
    st.markdown("<h2 class='center'>📊 Prediction Confidence</h2>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <p class='center'>
        <b>Normal:</b> {normal_prob:.1f}% &nbsp;&nbsp;&nbsp;
        <b>Abnormal:</b> {abnormal_prob:.1f}%
        </p>
        """,
        unsafe_allow_html=True
    )

    
    st.divider()
    st.markdown("<h2 class='center'>⚠️ Risk Explanation</h2>", unsafe_allow_html=True)

    reasons = []
    if prediction == 1:
        if deviation_abs > 10:
            reasons.append("Significant deviation from expected electricity usage")

        if usage_ratio > 1.5:
            reasons.append("Actual electricity consumption is much higher than expected")

        if load_utilization > 8:
            reasons.append("Electrical load utilization is unusually high")

        if appliance_score > 18:
            reasons.append("High appliance score contributes to elevated demand")

        if not reasons:
            reasons.append("Multiple moderate factors collectively contributed to abnormal usage")

    else:
        st.success("✅ Electricity usage is within normal operating limits")
        reasons.append("Usage pattern falls within normal operating limits")

    
    for r in reasons:
        st.markdown(f"<p class='center'>• {r}</p>",unsafe_allow_html=True)

    # ==============================
    # 🎯 User-Specific Impact (SHAP)
    # ==============================

    st.divider()
    st.markdown("<h2 class='center'>🎯 What Influenced Your Result</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        <p style='text-align: center; font-size: 14px; color: gray;'>
        These factors show which inputs most influenced the prediction.
        Higher percentage means stronger influence.
        </p>
        """,
        unsafe_allow_html=True
    )


    shap_values = explainer.shap_values(input_scaled)

    # ===== Robust handling for RandomForest (all SHAP versions) =====
    if isinstance(shap_values, list):
    # Old style → list per class
        user_shap = shap_values[1][0]

    elif len(shap_values.shape) == 3:
    # New style → (samples, features, classes)
        user_shap = shap_values[0, :, 1]

    else:
    # Fallback
        user_shap = shap_values[0]

    # Friendly feature names (MATCH USER UI)
    friendly_names = [
        "📍 Region",
        "🏢 Dwelling Type",
        "👨‍👩‍👧 Number of Occupants",
        "🏠 House Area (sqft)",
        "🔌 Appliance Score",
        "⚡ Connected Load (kW)",
        "🌡 Temperature (°C)",
        "💧 Humidity (%)",
        "📉 Usage Deviation (kWh)",
        "📊 Usage Ratio",
        "⚙️ Load Utilization"
    ]

    # Convert to percentage impact
    impact = user_shap.flatten()

    # ⭐ THEN normalize
    abs_impact = np.abs(impact)

    total = abs_impact.sum()
    if total == 0:
        impact_percent = abs_impact
    else:
        impact_percent = (abs_impact / total) * 100
      
    # Show top 6 most important for this user
    top_n = 6
    sorted_idx = np.argsort(impact_percent)[-top_n:]
    
    # Color bars by direction

    fig_shap = go.Figure(
        go.Bar(
            x=impact_percent[sorted_idx],
            y=np.array(friendly_names)[sorted_idx],
            orientation="h",
            text=[f"{v:.1f}%" for v in impact_percent[sorted_idx]],
            textposition="outside"
        )
    )

    fig_shap.update_layout(
        height=380,
        yaxis=dict(autorange="reversed"),
        margin=dict(l=160, r=40, t=30, b=30),
        xaxis_title="Impact on Prediction (%)"
    )

    st.plotly_chart(fig_shap, use_container_width=True)

    
    st.divider()

    st.markdown("<h2 style='text-align:center;'>🛠 Recommended Actions</h2>",unsafe_allow_html=True)

    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)

    if prediction == 1:
        actions = [
        "Reduce simultaneous use of high-power appliances",
        "Inspect appliances contributing to abnormal usage",
        "Optimize cooling and heating during peak hours",
        "Review connected electrical load capacity",
        "Upgrade to energy-efficient appliances"
        ]
    else:
        actions = [
        "Maintain current electricity usage pattern",
        "Monitor peak-hour consumption regularly",
        "Adopt energy-efficient usage practices"
        ]
    
    actions_html = "".join([f"<li>{a}</li>" for a in actions])
    st.markdown(
    f"""
    <div style="text-align:center;">
        <ul style="
            display:inline-block;
            text-align:left;
            list-style-position: inside;
            padding: 0;
            margin: 0;
        ">
            {actions_html}
        </ul>
    </div>
    """,
    unsafe_allow_html=True
    )




