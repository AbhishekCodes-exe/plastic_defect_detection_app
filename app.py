import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import matplotlib.pyplot as plt

# ============================================
# CUSTOM CSS (GLASSMORPHISM + ANIMATIONS)
# ============================================

page_bg = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: white;
    padding: 20px;
}

/* GLASS CARD */
.glass-card {
    background: rgba(255,255,255,0.12);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.3);
    box-shadow: 0px 8px 25px rgba(0,0,0,0.25);
    backdrop-filter: blur(10px);
    animation: fadeIn 1.3s ease;
}

/* HEADER TITLE */
.title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    margin-bottom: -5px;
    background: -webkit-linear-gradient(#ffffff, #d4d4d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* SUBTITLE */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cccccc;
}

/* FADE ANIMATION */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}

</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ============================================
# LOAD MODEL
# ============================================

MODEL_PATH = "plastic_defect_model.h5"
model = load_model(MODEL_PATH)

# UPDATE LABELS (must match your dataset)
LABELS = ["BlackDot", "BurnMark", "Flash", "FlowLines", "OK", "ShortShot"]

# ============================================
# SIDEBAR
# ============================================

st.sidebar.title("📌 Navigation")
st.sidebar.markdown("Use the menu below:")

page = st.sidebar.radio("Go to:", ["🏠 Home", "🔍 Detection", "ℹ️ About"])

# ============================================
# HOME PAGE
# ============================================

if page == "🏠 Home":
    st.markdown("<div class='title'>AI Defect Detection</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Plastic Injection Molding • Manufacturing Quality Control</div><br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass-card">
            <h3>🚀 Why this tool?</h3>
            <p>This application uses a Deep Learning model (MobileNetV2) to automatically detect defects 
            in plastic injection molded components. It helps in:</p>
            <ul>
                <li>Reducing manual inspection time</li>
                <li>Increasing accuracy & consistency</li>
                <li>Automating quality control pipelines</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================
# DETECTION PAGE
# ============================================

elif page == "🔍 Detection":

    st.markdown("<div class='title'>Defect Detection</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Upload a plastic component image</div><br>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📸 Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        img_resized = img.resize((224, 224))
        x = image.img_to_array(img_resized)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        with st.spinner("🔍 Analyzing Image..."):
            preds = model.predict(x)[0]
            index = np.argmax(preds)
            label = LABELS[index]
            confidence = float(preds[index])

        st.markdown("<br><div class='glass-card'>", unsafe_allow_html=True)
        st.subheader(f"🔎 Prediction: **{label}**")
        st.write(f"🔥 Confidence Score: **{confidence:.4f}**")

        # BAR CHART FOR CONFIDENCE
        fig, ax = plt.subplots(figsize=(8,3))
        ax.bar(LABELS, preds, color="#00e5ff")
        ax.set_ylabel("Confidence")
        ax.set_title("Prediction Probability Distribution")
        st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)

# ============================================
# ABOUT PAGE
# ============================================

elif page == "ℹ️ About":
    st.markdown("<div class='title'>About the Project</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass-card">
        <h3>📘 Project Overview</h3>
        <p>This project demonstrates how AI can be used for automated defect detection 
        in plastic injection molding. Built using:</p>

        - 🧠 MobileNetV2 Deep Learning Model  
        - 🖥️ TensorFlow & Keras  
        - 🎨 Streamlit UI  
        - 📊 Confidence Visualizations  

        <br>
        <h3>👨‍💻 Developer</h3>
        <p>Built by Abhishek — bringing AI to real manufacturing.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
