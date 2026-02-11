import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import sys
import io
import requests

# --- FORCED VENV PATH FIX ---
venv_site_packages = os.path.join(os.getcwd(), "venv", "Lib", "site-packages")
if venv_site_packages not in sys.path:
    sys.path.append(venv_site_packages)

# 1. Safe Import Logic
try:
    import mediapipe as mp
    mp_selfie = mp.solutions.selfie_segmentation
    IMPORT_SUCCESS = True
except Exception as e:
    IMPORT_SUCCESS = False

st.set_page_config(page_title="AI Pro Studio", layout="wide")

if not IMPORT_SUCCESS:
    st.error("MediaPipe is not initialized. Please ensure the venv is active.")
    st.stop()

# 2. Model Loader
@st.cache_resource
def load_model():
    return mp_selfie.SelfieSegmentation(model_selection=1)

selfie_segmentation = load_model()

st.title(" AI Background Studio Pro")

# --- SIDEBAR ---
st.sidebar.header("🌆 Background Gallery")

bg_category = st.sidebar.selectbox("Select Theme",
    ["Standard Effects", "Professional Office", "Nature & Scenery", "Sci-Fi & Cyberpunk", "Custom Upload"]
)

# Sub-options and Custom Upload Logic
bg_style = ""
custom_bg_file = None

if bg_category == "Standard Effects":
    bg_style = st.sidebar.selectbox("Effect", ["Blur Background", "Solid Color", "Digital Green Screen"])
elif bg_category == "Professional Office":
    bg_style = st.sidebar.selectbox("Office Type", ["Modern Glass Office", "Minimalist Studio", "Luxury Library", "Bed"])
elif bg_category == "Nature & Scenery":
    bg_style = st.sidebar.selectbox("Location", ["Tropical Beach", "Mountain Mist", "Autumn Forest"])
elif bg_category == "Sci-Fi & Cyberpunk":
    bg_style = st.sidebar.selectbox("World", ["Cyberpunk Street", "Deep Space Station", "Neon Grid", "no way"])
elif bg_category == "Custom Upload":
    bg_style = "Custom"
    custom_bg_file = st.sidebar.file_uploader("Upload your own Background", type=["jpg", "png", "jpeg"])

# Styling Controls
st.sidebar.divider()
st.sidebar.header("⚙️ Fine Tuning")
threshold = st.sidebar.slider("Edge Sharpness", 0.0, 1.0, 0.2, 0.05)
blur_val = st.sidebar.slider("Blur Strength", 5, 95, 35, 2) if "Blur" in bg_style else 0

# Handle Image Input (The Person)
mode = st.sidebar.radio("Input Source", ["Upload Photo", "Live Webcam"])
if mode == "Upload Photo":
    img_file = st.file_uploader("Choose a photo of a person", type=["jpg", "png", "jpeg"])
else:
    img_file = st.camera_input("Smile for the camera!")

BG_URLS = {
    "Modern Glass Office": "https://images.unsplash.com/photo-1497366216548-37526070297c?q=80&w=1000",
    "Luxury Library": "https://images.unsplash.com/photo-1507842217343-583bb7270b66?q=80&w=1000",
    "Tropical Beach": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?q=80&w=1000",
    "Mountain Mist": "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?q=80&w=1000",
    "Cyberpunk Street": "https://images.unsplash.com/photo-1605810230434-7631ac76ec81?q=80&w=1000",
    "Deep Space Station": "https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=1000",
    "no way": "https://plus.unsplash.com/premium_photo-1770723751148-89490a75e354?q=80&w=1170",
    "Bed": "https://plus.unsplash.com/premium_photo-1733864822156-f3cf26187fd9?q=80&w=1171"
}

if img_file:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    # AI Processing
    results = selfie_segmentation.process(image_rgb)
    mask = results.segmentation_mask
    condition = np.stack((mask,) * 3, axis=-1) > threshold

    # --- Background Selection Logic ---
    background = np.zeros(image.shape, dtype=np.uint8) # Default black

    if bg_style == "Blur Background":
        background = cv2.GaussianBlur(image, (blur_val, blur_val), 0)
    elif bg_style == "Solid Color":
        color = st.sidebar.color_picker("Pick Color", "#00FF00")
        c = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        background = np.full(image.shape, c[::-1], dtype=np.uint8)
    elif bg_style == "Digital Green Screen":
        background = np.full(image.shape, (0, 255, 0), dtype=np.uint8)
    
    # NEW CUSTOM UPLOAD LOGIC
    elif bg_category == "Custom Upload":
        if custom_bg_file is not None:
            custom_img = Image.open(custom_bg_file).convert("RGB")
            custom_img = custom_img.resize((w, h))
            background = cv2.cvtColor(np.array(custom_img), cv2.COLOR_RGB2BGR)
        else:
            st.warning("Please upload a background image in the sidebar!")
            background = np.zeros(image.shape, dtype=np.uint8)

    elif bg_style in BG_URLS:
        resp = requests.get(BG_URLS[bg_style], stream=True).raw
        bg_data = Image.open(resp).convert("RGB").resize((w, h))
        background = cv2.cvtColor(np.array(bg_data), cv2.COLOR_RGB2BGR)

    # Composite Merging
    
    output_image = np.where(condition, image, background)
    output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    # Display Results
    c1, c2 = st.columns(2)
    with c1:
        st.image(image_rgb, caption="Original Photo", use_container_width=True)
    with c2:
        st.image(output_rgb, caption="AI Studio Result", use_container_width=True)

        # Download Logic
        result_pil = Image.fromarray(output_rgb)
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        st.download_button("💾 Download Studio Photo", data=buf.getvalue(), 
                           file_name="studio_portrait.png", mime="image/png")

st.divider()
st.caption("AI Studio v2.0 | Advanced Custom Background Support")