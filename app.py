import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import plotly.graph_objects as go

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="AI Plastic Defect Detector",
    layout="wide",
    page_icon="🧠"
)

# ============================================
# CUSTOM CSS (GLASSMORPHISM, MODERN UI)
# ============================================
custom_css = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif !important;
}

.stApp {
    background: linear-gradient(135deg, #141e30, #243b55);
    padding: 20px;
    color: white;
}

.glass-card {
    background: rgba(255, 255, 255, 0.12);
    padding: 25px;
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.25);
    backdrop-filter: blur(10px);
    box-shadow: 0px 8px 25px rgba(0,0,0,0.25);
    animation: fadeIn 1s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0px); }
}

.upload-box {
    border: 2px dashed #00eaff;
    border-radius: 15px;
    padding: 30px;
    text-align: center;
    background: rgba(255,255,255,0.05);
}

.upload-box:hover {
    background: rgba(255,255,255,0.12);
    border-color: white;
}

</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ============================================
# LOAD MODEL
# ============================================
MODEL_PATH = "plastic_defect_model.h5"
model = load_model(MODEL_PATH)

LABELS = ["BlackDot", "BurnMark", "Flash", "FlowLines", "OK", "ShortShot"]

# Save prediction history
if "history" not in st.session_state:
    st.session_state.history = []

# ============================================
# SIDEBAR
# ============================================
st.sidebar.title("📌 Navigation")

page = st.sidebar.radio("Menu", ["🏠 Home", "🔍 Detection", "📜 History", "ℹ️ About"])

st.sidebar.markdown("---")
st.sidebar.write("Model: MobileNetV2")
st.sidebar.write(f"Classes: {len(LABELS)}")

# ============================================
# HOME PAGE
# ============================================
if page == "🏠 Home":
    st.markdown("<h1 style='font-weight:800;'>AI Plastic Defect Detection</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#d7d7d7;'>Deep Learning Powered • Quality Inspection System</h4>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass-card">
        <h3>🚀 What this app does</h3>
        <p>This system automatically detects visual defects in plastic injection molded parts using a deep learning model.</p>

        ### Benefits:
        - Automated Quality Check  
        - Consistent & Accurate  
        - Zero Human Error  
        - Industry Ready AI Solution  
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================
# DETECTION PAGE
# ============================================
elif page == "🔍 Detection":

    st.markdown("<h1 style='font-weight:800;'>Defect Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#d0d0d0;'>Upload an image to analyze</p>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload image(s) of plastic parts",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:

            img = Image.open(uploaded_file).convert("RGB")
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.image(img, caption="Uploaded Image", use_column_width=True)

            # Preprocess
            img_resized = img.resize((224, 224))
            x = image.img_to_array(img_resized)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            preds = model.predict(x)[0]
            index = np.argmax(preds)
            label = LABELS[index]
            confidence = float(preds[index])

            # Save history
            st.session_state.history.append((uploaded_file.name, label, confidence))

            # Donut Chart
            fig = go.Figure(data=[go.Pie(
                labels=["Confidence", "Remaining"],
                values=[confidence, 1-confidence],
                hole=.7,
                marker_colors=["#00eaff", "#202020"]
            )])
            fig.update_layout(showlegend=False, height=300)

            st.markdown(f"<h2 style='color:#00eaff;'>🔎 Prediction: {label}</h2>", unsafe_allow_html=True)
            st.write(f"🔥 Confidence Score: **{confidence:.4f}**")

            colA, colB = st.columns(2)
            with colA:
                st.plotly_chart(fig, use_container_width=True)

            with colB:
                # Bar chart
                fig2 = go.Figure([go.Bar(
                    x=LABELS,
                    y=preds,
                    marker_color="#00eaff"
                )])
                fig2.update_layout(height=300, title="Confidence Distribution")
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# HISTORY PAGE
# ============================================
elif page == "📜 History":
    st.markdown("<h1 style='font-weight:800;'>Prediction History</h1>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    if len(st.session_state.history) == 0:
        st.write("No predictions made yet.")
    else:
        for item in st.session_state.history:
            st.write(f"📁 **{item[0]}** → 🔍 {item[1]} ({item[2]:.4f})")

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# ABOUT PAGE
# ============================================
elif page == "ℹ️ About":
    st.markdown("<h1 style='font-weight:800;'>About This Project</h1>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass-card">
        <h3>📘 Overview</h3>
        <p>This AI-powered system detects defects in plastic injection molded parts 
        using deep learning (MobileNetV2). Designed for manufacturing companies, QC teams, 
        and industrial automation use cases.</p>

        <h3>👨‍💻 Developer</h3>
        <p>Built by Abhishek — AI Developer</p>

        </div>
        """,
        unsafe_allow_html=True
    )
