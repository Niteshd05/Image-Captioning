import streamlit as st
from roboflow import Roboflow
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import tempfile
import os
import io
import requests
import base64

# ------------------------------------------------
# Streamlit Configuration
# ------------------------------------------------
st.set_page_config(page_title="Smart Vision Assistant (BLIP-2)", page_icon="🧠")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------
# Model Loading (Cached)
# ------------------------------------------------
@st.cache_resource
def load_models():
    # Roboflow currency detection
    st.info("🔄 Loading Roboflow model...")
    rf = Roboflow(api_key="2po24idSl5m93Vfr6ZtF")
    project = rf.workspace().project("indian-currency-detection-elfyf")
    currency_model = project.version(1).model

    # BLIP-2 model (Flan-T5-XL)
    st.info("🔄 Loading BLIP-2 model...")
    model_id = "Salesforce/blip2-flan-t5-xl"
    processor = Blip2Processor.from_pretrained(model_id, use_fast=True)
    caption_model = Blip2ForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.float16
    ).to(DEVICE)

    return currency_model, processor, caption_model


currency_model, processor, caption_model = load_models()

# ------------------------------------------------
# ElevenLabs Text-to-Speech
# ------------------------------------------------
ELEVENLABS_API_KEY = st.secrets.get("ELEVENLABS_API_KEY", "")
VOICE_ID = "Rachel"  # Female voice (Free-tier available)

def speak_currency(text):
    """Generate speech with ElevenLabs and autoplay."""
    try:
        if ELEVENLABS_API_KEY:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": ELEVENLABS_API_KEY
            }
            payload = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {"stability": 0.4, "similarity_boost": 0.8}
            }
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                audio_bytes = response.content
            else:
                raise Exception(f"ElevenLabs request failed: {response.text}")
        else:
            raise Exception("Missing ElevenLabs API key.")
    except Exception as e:
        st.warning(f"TTS error: {e}")
        return

    # Autoplay in browser
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
        <audio autoplay="true">
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

# ------------------------------------------------
# Caption Generation (BLIP-2)
# ------------------------------------------------
def generate_caption(image: Image.Image):
    prompt = (
        "Describe this image in detailed and visually rich language, "
        "focusing on people, objects, context, and emotions."
    )
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)
    output = caption_model.generate(**inputs, max_new_tokens=100)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption.capitalize()

# ------------------------------------------------
# Currency Detection (Roboflow)
# ------------------------------------------------
def detect_currency(image_path):
    result = currency_model.predict(image_path)
    annotated_path = image_path.replace(".jpg", "_annotated.jpg")
    result.save(annotated_path)
    preds = result.json().get("predictions", [])
    currency_detected = preds[-1]["class"] if preds else None
    return currency_detected, annotated_path

# ------------------------------------------------
# Streamlit UI
# ------------------------------------------------
st.title("🧠 Smart Vision Assistant (BLIP-2 + ElevenLabs)")
st.write("Detect Indian currency and describe surroundings with rich visual captions.")
st.markdown("---")

# Upload Section
st.subheader("📁 Upload Image(s)")
uploaded_files = st.file_uploader(
    "Upload one or more images of currency or surroundings",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.info(f"Processing: {uploaded_file.name}")
        file_bytes = uploaded_file.read()

        with Image.open(io.BytesIO(file_bytes)) as image:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                image.save(tmp.name)
                temp_path = tmp.name

            with st.spinner("🧠 Generating description..."):
                caption = generate_caption(image)

        with st.spinner("💵 Detecting currency..."):
            currency, annotated_path = detect_currency(temp_path)

        st.image(annotated_path, caption="Annotated Result", use_container_width=True)

        if currency:
            message = f"I see an Indian {currency} note. {caption}"
        else:
            message = f"No currency detected. {caption}"

        st.success(message)
        speak_currency(message)

        # Cleanup
        for f in [temp_path, annotated_path]:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except PermissionError:
                pass

st.markdown("---")

# ------------------------------------------------
# Webcam Capture Section (Streamlit supported)
# ------------------------------------------------
st.subheader("📷 Capture via Webcam")

img_file = st.camera_input("Capture an image from your webcam")

if img_file:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img = Image.open(img_file)
        img.save(tmp.name)
        temp_file = tmp.name

    with st.spinner("🧠 Generating description..."):
        caption = generate_caption(img)

    with st.spinner("💵 Detecting currency..."):
        currency, annotated_path = detect_currency(temp_file)

    st.image(annotated_path, caption="Webcam Annotated Image", use_container_width=True)

    if currency:
        message = f"I can see an Indian {currency} note. {caption}"
    else:
        message = f"No recognizable currency detected. {caption}"

    st.success(message)
    speak_currency(message)

    for f in [temp_file, annotated_path]:
        try:
            if os.path.exists(f):
                os.remove(f)
        except PermissionError:
            pass
