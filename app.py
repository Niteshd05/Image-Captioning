import streamlit as st
from roboflow import Roboflow
from gtts import gTTS
from PIL import Image
import io
import tempfile
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import os, base64, requests

# -------------------------------
# Streamlit Config
# -------------------------------
st.set_page_config(page_title="Smart Vision Assistant", page_icon="🧠")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# ElevenLabs Config
# -------------------------------
ELEVENLABS_API_KEY = st.secrets.get("ELEVENLABS_API_KEY", None)
VOICE_ID = "Elli"  # Free-tier female voice

# -------------------------------
# Model Loader (Cached)
# -------------------------------
@st.cache_resource
def load_models():
    # Roboflow currency detection
    rf = Roboflow(api_key="2po24idSl5m93Vfr6ZtF")
    project = rf.workspace().project("indian-currency-detection-elfyf")
    currency_model = project.version(1).model

    # BLIP image captioning
    caption_model_id = "Salesforce/blip-image-captioning-large"
    processor = BlipProcessor.from_pretrained(caption_model_id)
    caption_model = BlipForConditionalGeneration.from_pretrained(caption_model_id).to(DEVICE)

    return currency_model, processor, caption_model


currency_model, processor, caption_model = load_models()

# -------------------------------
# ElevenLabs + gTTS Speech Function
# -------------------------------
def speak_currency(text: str):
    """Generate audio using ElevenLabs (fallback: gTTS) and autoplay."""
    try:
        # Try ElevenLabs
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
                raise Exception("ElevenLabs API Error")
        else:
            raise Exception("No ElevenLabs key found")

    except Exception:
        # Fallback to Google TTS
        tts = gTTS(text)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        audio_bytes = audio_fp.read()

    # Autoplay audio
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
        <audio autoplay="true">
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

# -------------------------------
# Generate Caption (BLIP)
# -------------------------------
def generate_caption(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    out = caption_model.generate(**inputs, max_length=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption.capitalize()

# -------------------------------
# Detect Currency (Roboflow)
# -------------------------------
def detect_currency(image_path):
    result = currency_model.predict(image_path)
    annotated_path = image_path.replace(".jpg", "_annotated.jpg")
    result.save(annotated_path)
    preds = result.json().get("predictions", [])
    currency_detected = preds[-1]["class"] if preds else None
    return currency_detected, annotated_path

# -------------------------------
# UI Layout
# -------------------------------
st.title("🧠 Smart Vision Assistant for the Visually Impaired")
st.write("Detect Indian currency, describe surroundings, and read results aloud.")
st.markdown("---")

# -------------------------------
# Upload Image(s)
# -------------------------------
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

        for f in [temp_path, annotated_path]:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except PermissionError:
                st.warning(f"Could not delete temporary file {f}")

st.markdown("---")

# -------------------------------
# Webcam Section (Using Streamlit camera_input)
# -------------------------------
st.subheader("📷 Capture via Webcam")

camera_image = st.camera_input("Take a photo")

if camera_image is not None:
    # Save captured frame temporarily
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(camera_image.getbuffer())
        temp_file = tmp.name

    # Open image safely
    with Image.open(temp_file) as image:
        with st.spinner("🧠 Generating description..."):
            caption = generate_caption(image)

    # Detect currency
    with st.spinner("💵 Detecting currency..."):
        currency, annotated_path = detect_currency(temp_file)

    st.image(annotated_path, caption="Webcam Annotated Image", use_container_width=True)

    # Message + TTS
    if currency:
        message = f"I can see an Indian {currency} note."
    else:
        message = f"{caption}"

    st.success(message)
    speak_currency(message)

    # Cleanup temp files
    for f in [temp_file, annotated_path]:
        try:
            if os.path.exists(f):
                os.remove(f)
        except PermissionError:
            st.warning(f"Could not delete temporary file {f}")

