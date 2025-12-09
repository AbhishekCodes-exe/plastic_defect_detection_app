import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import json
import plotly.graph_objects as go
from streamlit_lottie import st_lottie

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="AI Plastic Defect Detector",
    page_icon="🧠",
    layout="wide",
)

# ============================================
# LOAD LOTTIE ANIMATION
# ============================================

def load_lottie(path):
    with open(path, "r") as f:
        return json.load(f)

lottie_ai = load_lottie("animations/ai.json")  # Replace with your Lottie file

# ============================================
# CUSTOM UI CSS
# ============================================

custom_css = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif !important;
}

.stApp {
    background: linear-gradient(135deg, #141e30, #243b55);
    padding: 15px;
}

.glass-card {
    background: rgba(255, 255, 255, 0.13);
    border-radius: 18px;
    padding: 25px;
    border: 1px solid rgba(255,255,255,0.25);
    backdrop-filter: blur(10px);
    box-shadow: 0px 4px 25px rgba(0,0,0,0.25);
    animation: fadeIn 1.5s ease;
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(15px);
    }
    to {
        opacity: 1;
        transform: translateY(0px);
    }
}

.upload-box {
    border: 2px dashed #00eaff;
    border-radius: 15px;
    padding: 35px;
    text-align: center;
    background: rgba(255,255,255,0.05);
    transition: 0.3s;
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

# EDIT YOUR LABELS HERE
LABELS = ["BlackDot", "BurnMark", "Flash", "FlowLines", "OK", "ShortShot"]

# For prediction history
if "history" not in st.session_state:
    st.session_state.history = []

# ============================================
# SIDEBAR
# ============================================

st.sidebar.image("logo.png", width=180, caption="Your Company Logo")

st.sidebar.title("⚙️ Settings")
theme = st.sidebar.radio("Theme Mode:", ["Dark", "Light"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Stats")
st.sidebar.write("Model: MobileNetV2")
st.sidebar.write(f"Classes: {len(LABELS)}")

# ============================================
# HEADER
# ============================================

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("<h1 style='color:white;font-weight:800;'>AI Plastic Defect Detection</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#dcdcdc;'>Automated Quality Inspection • Deep Learning Powered</h4>", unsafe_allow_html=True)

with col2:
    st_lottie(lottie_ai, height=200)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================
# DRAG & DROP UPLOADER
# ============================================

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<h3 style="color:white;">📸 Upload Image(s)</h3>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Drag & Drop or Browse",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

st.markdown('</div>', unsafe_allow_html=True)

if uploaded_files:
    for uploaded_file in uploaded_files:

        img = Image.open(uploaded_file).convert("RGB")

        st.markdown('<br><div class="glass-card">', unsafe_allow_html=True)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        img_resized = img.resize((224, 224))
        x = image.img_to_array(img_resized)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)[0]
        index = np.argmax(preds)
        label = LABELS[index]
        confidence = float(preds[index])

        # SAVE HISTORY
        st.session_state.history.append((uploaded_file.name, label, confidence))

        # Donut Chart (Plotly)
        fig = go.Figure(data=[go.Pie(
            labels=["Confidence", "Remaining"],
            values=[confidence, 1-confidence],
            hole=.7,
            marker_colors=["#00eaff", "#1f1f1f"])
        ])
        fig.update_layout(showlegend=False, height=300)

        st.markdown("<h3 style='color:white;'>🔍 Prediction Result</h3>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='color:#00eaff;'>🟦 {label}</h2>", unsafe_allow_html=True)
        st.write(f"**Confidence:** {confidence:.4f}")

        colA, colB = st.columns([1, 1])
        with colA:
            st.plotly_chart(fig, use_container_width=True)

        # Bar chart
        with colB:
            fig2 = go.Figure([go.Bar(
                x=LABELS,
                y=preds,
                marker_color="#00eaff"
            )])
            fig2.update_layout(title="Confidence Distribution", height=300)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# PREDICTION HISTORY
# ============================================

st.markdown("<br><div class='glass-card'>", unsafe_allow_html=True)
st.markdown("### 📜 Prediction History")

if len(st.session_state.history) == 0:
    st.write("No predictions yet.")
else:
    for item in st.session_state.history:
        st.write(f"📁 **{item[0]}** → 🔎 {item[1]} ({item[2]:.2f})")

st.markdown("</div>", unsafe_allow_html=True)
