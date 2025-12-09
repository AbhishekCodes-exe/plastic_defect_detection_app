import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image
import base64

# -----------------------------------------
# Custom CSS for a beautiful modern UI
# -----------------------------------------
page_bg = """
<style>
    .stApp {
        background: linear-gradient(135deg, #0F2027, #203A43, #2C5364);
        padding: 20px;
    }
    .title {
        font-size: 42px;
        font-weight: 800;
        color: #ffffff;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 20px;
        color: #d9d9d9;
        text-align: center;
        margin-bottom: 25px;
    }
    .result-box {
        padding: 20px;
        border-radius: 15px;
        background-color: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.3);
        backdrop-filter: blur(10px);
        margin-top: 20px;
    }
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# -----------------------------------------
# Load Model
# -----------------------------------------
MODEL_PATH = "plastic_defect_model.h5"
model = load_model(MODEL_PATH)

# UPDATE LABELS AS PER YOUR DATASET
LABELS = ["BlackDot", "BurnMark", "Flash", "FlowLines", "OK", "ShortShot"]

# -----------------------------------------
# UI Title
# -----------------------------------------
st.markdown('<div class="title">Plastic Defect Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-based Inspection | Powered by Deep Learning (MobileNetV2)</div>', unsafe_allow_html=True)

# -----------------------------------------
# Upload Image
# -----------------------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_resized = img.resize((224, 224))
    x = image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)[0]
    index = np.argmax(preds)
    label = LABELS[index]
    confidence = float(preds[index])

    # NICE RESULT UI BOX
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.subheader(f"🔍 Prediction: **{label}**")
    st.write(f"🔥 Confidence Score: **{confidence:.2f}**")
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Please upload an image to begin detection.")
