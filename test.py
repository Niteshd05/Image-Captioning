import streamlit as st
from roboflow import Roboflow
import pyttsx3
from PIL import Image
import io
import tempfile
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
from pathlib import Path
import os
import time
import numpy as np
import base64

# -------------------------------
# Streamlit App Config
# -------------------------------
st.set_page_config(
    page_title="Smart Vision Assistant",
    page_icon="🧠",
    layout="wide"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Load Models (Cached)
# -------------------------------
@st.cache_resource
def load_models():
    models = {}

    # Roboflow currency detection
    try:
        rf = Roboflow(api_key="2po24idSl5m93Vfr6ZtF")
        project = rf.workspace().project("indian-currency-detection-elfyf")
        models['currency'] = project.version(1).model
        st.sidebar.success("✅ Currency model loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Currency model error: {e}")

    # BLIP captioning
    try:
        caption_model_id = "Salesforce/blip-image-captioning-large"
        processor = BlipProcessor.from_pretrained(caption_model_id)
        caption_model = BlipForConditionalGeneration.from_pretrained(caption_model_id)
        caption_model = caption_model.to(DEVICE).eval()

        models['processor'] = processor
        models['caption'] = caption_model
        st.sidebar.success("✅ Caption model loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Caption model error: {e}")

    # YOLO models for navigation
    nav_paths = {
        'crosswalk': 'best.pt',
        'door': 'doors.pt'
    }

    for name, path in nav_paths.items():
        try:
            if Path(path).exists():
                models[name] = YOLO(path)
                st.sidebar.success(f"✅ {name.capitalize()} model loaded")
            else:
                st.sidebar.warning(f"⚠️ {name.capitalize()} model not found: {path}")
        except Exception as e:
            st.sidebar.error(f"❌ {name.capitalize()} model error: {e}")

    return models

models = load_models()

# -------------------------------
# Text-to-Speech
# -------------------------------
def speak(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        voices = engine.getProperty("voices")
        for v in voices:
            if "female" in v.name.lower():
                engine.setProperty("voice", v.id)
                break
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        st.error(f"TTS Error: {e}")

# -------------------------------
# Generate Caption
# -------------------------------
def generate_caption(image: Image.Image):
    if 'processor' not in models or 'caption' not in models:
        return "Caption model not available."

    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')

        inputs = models['processor'](images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = models['caption'].generate(**inputs, max_length=50)
        caption = models['processor'].decode(out[0], skip_special_tokens=True)
        return caption.capitalize()
    except Exception as e:
        st.warning(f"Caption generation error: {e}")
        return "Scene description unavailable."

# -------------------------------
# Detect Currency
# -------------------------------
def detect_currency(image_path):
    if 'currency' not in models:
        return None, image_path

    result = models['currency'].predict(image_path)
    annotated_path = image_path.replace(".jpg", "_annotated.jpg")
    result.save(annotated_path)
    preds = result.json().get("predictions", [])
    currency_detected = preds[-1]["class"] if preds else None
    return currency_detected, annotated_path

# -------------------------------
# Detect Navigation Objects
# -------------------------------
def detect_navigation(image_path, confidence=0.0):
    if not any(k in models for k in ['crosswalk', 'door']):
        return None, image_path, []

    import cv2
    frame = cv2.imread(image_path)
    annotated_frame = frame.copy()
    detections = []

    colors = {'crosswalk': (0, 255, 0), 'door': (255, 128, 0)}

    for model_name in ['crosswalk', 'door']:
        if model_name not in models:
            continue

        results = models[model_name].predict(frame, conf=confidence, verbose=False)
        color = colors.get(model_name, (0, 150, 255))

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                class_name = result.names[cls]
                detections.append({
                    'type': model_name,
                    'class': class_name,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2)
                })
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{model_name}: {class_name} {conf:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    annotated_path = image_path.replace(".jpg", "_nav_annotated.jpg")
    cv2.imwrite(annotated_path, annotated_frame)
    return annotated_frame, annotated_path, detections

# -------------------------------
# Format Navigation Message
# -------------------------------
def format_navigation_message(detections, caption):
    if not detections:
        return f"No navigation objects detected. {caption}"

    doors = sum(1 for d in detections if d['type'] == 'door')
    crosswalks = sum(1 for d in detections if d['type'] == 'crosswalk')
    parts = []
    if doors > 0:
        parts.append(f"{doors} door{'s' if doors > 1 else ''}")
    if crosswalks > 0:
        parts.append(f"{crosswalks} crosswalk{'s' if crosswalks > 1 else ''}")
    return f"I can see {', '.join(parts)}. {caption}"

# -------------------------------
# UI Layout
# -------------------------------
st.title("🧠 Smart Vision Assistant for the Visually Impaired")

mode = st.radio("Select Mode:", ["💵 Currency Detection", "🚶 Navigation Assistant"], horizontal=True)
st.markdown("---")

# -------------------------------
# Currency Detection Mode
# -------------------------------
if mode == "💵 Currency Detection":
    st.subheader("💵 Currency Detection Mode")
    st.write("Detect Indian currency notes and describe surroundings.")

    uploaded_files = st.file_uploader("Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.info(f"Processing: {uploaded_file.name}")
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(uploaded_file.read())
                temp_path = tmp.name

            with Image.open(temp_path) as image:
                caption = generate_caption(image)
                currency, annotated_path = detect_currency(temp_path)
                st.image(annotated_path, caption="Result", use_container_width=True)

            message = f"I see an Indian {currency} note." if currency else caption
            st.success(message)
            speak(message)

            for f in [temp_path, annotated_path]:
                if os.path.exists(f): os.remove(f)

    st.markdown("### 📷 Capture via Webcam")
    camera_image = st.camera_input("Take a photo for currency detection")
    if camera_image is not None:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(camera_image.getbuffer())
            temp_path = tmp.name

        with Image.open(temp_path) as image:
            caption = generate_caption(image)
            currency, annotated_path = detect_currency(temp_path)
            st.image(annotated_path, caption="Camera Result", use_container_width=True)

        message = f"I can see an Indian {currency} note." if currency else caption
        st.success(message)
        speak(message)

        for f in [temp_path, annotated_path]:
            if os.path.exists(f): os.remove(f)

# -------------------------------
# Navigation Assistant Mode
# -------------------------------
else:
    st.subheader("🚶 Navigation Assistant Mode")
    st.write("Detect doors and crosswalks to assist navigation.")
    confidence = st.slider("Detection Confidence", 0.0, 1.0, 0.5, 0.05)

    uploaded_files = st.file_uploader("Upload navigation images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.info(f"Processing: {uploaded_file.name}")
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(uploaded_file.read())
                temp_path = tmp.name

            with Image.open(temp_path) as image:
                caption = generate_caption(image)
                _, annotated_path, detections = detect_navigation(temp_path, confidence)
                st.image(annotated_path, caption="Navigation Result", use_container_width=True)

            message = format_navigation_message(detections, caption)
            st.success(message)
            speak(message)

            for f in [temp_path, annotated_path]:
                if os.path.exists(f): os.remove(f)

    st.markdown("### 📷 Capture via Webcam")
    camera_image = st.camera_input("Take a photo for navigation detection")
    if camera_image is not None:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(camera_image.getbuffer())
            temp_path = tmp.name

        with Image.open(temp_path) as image:
            caption = generate_caption(image)
            _, annotated_path, detections = detect_navigation(temp_path, confidence)
            st.image(annotated_path, caption="Camera Navigation Result", use_container_width=True)

        message = format_navigation_message(detections, caption)
        st.success(message)
        speak(message)

        for f in [temp_path, annotated_path]:
            if os.path.exists(f): os.remove(f)

st.markdown("---")
st.markdown("**Note**: Ensure model files (`best.pt`, `doors.pt`) are present in the same directory.")
