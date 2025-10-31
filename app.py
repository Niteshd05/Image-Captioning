import streamlit as st
from roboflow import Roboflow
import pyttsx3
from PIL import Image
import io
import cv2
import tempfile
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import os, time

# -------------------------------
# Streamlit App Config
# -------------------------------
st.set_page_config(page_title="Smart Vision Assistant", page_icon="🧠")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Load Models (Cached)
# -------------------------------
@st.cache_resource
def load_models():
    # Roboflow currency detection
    rf = Roboflow(api_key="2po24idSl5m93Vfr6ZtF")
    project = rf.workspace().project("indian-currency-detection-elfyf")
    currency_model = project.version(1).model

    # BLIP captioning
    caption_model_id = "Salesforce/blip-image-captioning-large"
    processor = BlipProcessor.from_pretrained(caption_model_id)
    caption_model = BlipForConditionalGeneration.from_pretrained(caption_model_id).to(DEVICE)

    return currency_model, processor, caption_model

currency_model, processor, caption_model = load_models()

# -------------------------------
# Text-to-Speech
# -------------------------------
def speak(text):
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        for v in voices:
            if "female" in v.name.lower():
                engine.setProperty("voice", v.id)
                break
        engine.setProperty("rate", 175)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        st.error(f"TTS Error: {e}")

# -------------------------------
# Generate Caption
# -------------------------------
def generate_caption(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    out = caption_model.generate(**inputs, max_length=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption.capitalize()

# -------------------------------
# Detect Currency
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
# Image Upload Section
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

        # Use context manager to avoid file locks
        with Image.open(io.BytesIO(file_bytes)) as image:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                image.save(tmp.name)
                temp_path = tmp.name

            # Generate caption
            with st.spinner("🧠 Generating description..."):
                caption = generate_caption(image)

        # Detect currency
        with st.spinner("💵 Detecting currency..."):
            currency, annotated_path = detect_currency(temp_path)

        st.image(annotated_path, caption="Annotated Result", use_container_width=True)

        # Prepare message and speak
        if currency:
            message = f"I see an Indian {currency} note. {caption}"
        else:
            message = f"No currency detected. {caption}"

        st.success(message)
        speak(message)

        # Safe cleanup
        for f in [temp_path, annotated_path]:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except PermissionError:
                st.warning(f"Could not delete temporary file {f}")

st.markdown("---")

# -------------------------------
# Webcam Section
# -------------------------------
st.subheader("📷 Capture via Webcam")

if st.button("Start 5-Second Webcam Capture"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not access webcam.")
    else:
        st.info("Capturing frames for 5 seconds...")
        start_time = time.time()
        last_frame = None
        while time.time() - start_time < 5:
            ret, frame = cap.read()
            if not ret:
                break
            last_frame = frame.copy()
        cap.release()
        st.success("Capture complete!")

        if last_frame is not None:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                cv2.imwrite(tmp.name, last_frame)
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
                message = f"I can see an Indian {currency} note. {caption}"
            else:
                message = f"No recognizable currency detected. {caption}"

            st.success(message)
            speak(message)

            # Cleanup temp files
            for f in [temp_file, annotated_path]:
                try:
                    if os.path.exists(f):
                        os.remove(f)
                except PermissionError:
                    st.warning(f"Could not delete temporary file {f}")
