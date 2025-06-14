import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
import json
import torch
from transformers import CLIPProcessor, CLIPModel

# 🌿 Page config
st.set_page_config(page_title="🌿 Leaf Disease Detector", layout="wide")

# Load CNN model
MODEL_PATH = "model.h5"
if not os.path.exists(MODEL_PATH):
    st.info("📦 Downloading model...")
    gdown.download("https://drive.google.com/uc?id=1zcTv5D-w6caJwoOYPBht_VpsuTt9jCd5", MODEL_PATH, quiet=False)
model = tf.keras.models.load_model(MODEL_PATH)

# Load class names
with open("class_indices.json", "r") as f:
    index_to_class = json.load(f)
class_names = [index_to_class[str(i)] for i in range(len(index_to_class))]

# Load remedies
with open("remedies.json", "r") as f:
    remedies = json.load(f)

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")  # no .to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = torch.device("cpu")

# CLIP fallback prompts
clip_prompts = [
    "a healthy guava leaf",
    "a guava leaf with black fungus",
    "a guava leaf with yellow spots",
    "a diseased guava leaf",
    "a healthy mango leaf",
    "a mango leaf with black patches",
    "a mango leaf with curled edges",
    "a mango leaf with fungal infection",
    "a healthy jackfruit leaf",
    "a jackfruit leaf with holes",
    "a jackfruit leaf with rust spots",
    "a diseased jackfruit leaf",
    "a papaya leaf with yellowing",
    "a papaya leaf with virus infection",
    "a papaya leaf with leaf curl disease",
    "a banana leaf with rust disease",
    "a banana leaf with insect damage",
    "a banana leaf with black streaks",
    "a healthy banana leaf",
    "a neem leaf with powdery mildew",
    "a neem leaf with yellowing",
    "a neem leaf with bacterial blight",
    "a healthy neem leaf",
    "a betel leaf with rot",
    "a betel leaf with dryness",
    "a betel leaf with fungal infection",
    "a healthy betel leaf",
    "a hibiscus leaf with white patches",
    "a diseased hibiscus leaf",
    "a hibiscus leaf with holes",
    "a money plant leaf with yellowing",
    "a money plant leaf with root rot",
    "a healthy money plant leaf",
    "a tulsi leaf with black fungus",
    "a tulsi leaf with virus",
    "a tulsi leaf with mildew",
    "a healthy tulsi leaf",
    "a cotton leaf with curling",
    "a cotton leaf with powdery mildew",
    "a cotton leaf with bacterial blight",
    "a rice leaf with leaf blast",
    "a rice leaf with sheath blight",
    "a rice leaf with tungro virus",
    "a wheat leaf with rust",
    "a wheat leaf with powdery mildew",
    "a wheat leaf with leaf blight",
    "a barley leaf with fungal spots",
    "a barley leaf with bacterial stripe",
    "a sunflower leaf with downy mildew",
    "a sunflower leaf with rust",
    "a sunflower leaf with necrosis",
    "a brinjal leaf with shot hole",
    "a brinjal leaf with fungal infection",
    "a tomato leaf with bacterial spot",
    "a tomato leaf with late blight",
    "a tomato leaf with fungal infection",
    "a tomato leaf with virus",
    "a corn leaf with common rust",
    "a corn leaf with gray leaf spot",
    "a healthy corn leaf",
    "a grape leaf with black rot",
    "a grape leaf with measles",
    "a healthy grape leaf",
    "a potato leaf with early blight",
    "a potato leaf with late blight",
    "a potato leaf with curling",
    "a squash leaf with powdery mildew",
    "a pepper leaf with bacterial spot",
    "a pepper leaf with insect bites",
    "a strawberry leaf with scorch",
    "a strawberry leaf with holes",
    "a strawberry leaf with powdery mildew",
    "a peach leaf with bacterial spot",
    "a peach leaf with blight",
    "a healthy tropical leaf",
    "a tropical leaf with virus",
    "a tropical leaf with yellowing",
    "a fig leaf with rust",
    "a fig leaf with mildew",
    "a healthy fig leaf",
    "a drumstick leaf with yellow patches",
    "a moringa leaf with insect damage",
    "a curry leaf with leaf spot",
    "a curry leaf with drying",
    "a curry leaf with necrosis",
    "a tamarind leaf with fungus",
    "a tamarind leaf with brown edges",
    "a bamboo leaf with rust",
    "a bamboo leaf with yellowing",
    "a aloe vera leaf with fungal rot",
    "a aloe vera leaf with bacterial blight",
    "a cactus leaf with scarring",
    "a cactus leaf with dryness",
    "a tulip leaf with yellow spots",
    "a tulip leaf with blight",
    "a rose leaf with black spot",
    "a rose leaf with powdery mildew",
    "a marigold leaf with leaf miner",
    "a marigold leaf with insect bite",
    "a leaf infected by fungus",
    "a leaf with unknown disease",
    "a diseased garden leaf"
]
# PlantVillage trusted labels
trusted_labels = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy", "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy", "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight",
    "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight",
    "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

# UI
st.title("🌿 Leaf Disease Detector (CNN + CLIP Hybrid)")
uploaded_file = st.file_uploader("📷 Upload a leaf image", type=["jpg", "jpeg", "png"])
force_fallback = st.checkbox("⚙️ Force fallback to CLIP", value=False)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Uploaded Leaf", use_column_width=True)

    # CNN prediction
    resized_img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(resized_img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    confidence = float(np.max(prediction)) * 100
    predicted_index = int(np.argmax(prediction))
    predicted_class = class_names[predicted_index]

    st.markdown(f"🔍 CNN Prediction: `{predicted_class}` ({confidence:.2f}%)")

    # Trust logic
    is_trusted = confidence > 80 and any(
        predicted_class.strip().lower() in label.lower() or label.lower() in predicted_class.strip().lower()
        for label in trusted_labels
    )
    if is_trusted and not predicted_class.lower().startswith((
        "tomato", "apple", "potato", "grape", "corn", "pepper", "soybean",
        "orange", "strawberry", "cherry", "peach", "squash"
    )):
        is_trusted = False

    if not is_trusted or force_fallback:
        st.warning("⚠️ Using CLIP fallback due to untrusted or forced condition...")
        inputs = clip_processor(text=clip_prompts, images=img, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).numpy().flatten()
        top_indices = probs.argsort()[-3:][::-1]

        st.subheader("🧠 CLIP Top Matches")
        for i in top_indices:
            label = clip_prompts[i]
            score = probs[i] * 100
            st.markdown(f"- 🟢 **{label}** — `{score:.2f}%`")
    else:
        st.subheader("🧠 CNN Model Prediction")
        readable_class = predicted_class.replace("_", " ")
        st.success(f"{readable_class} ({confidence:.2f}%)")

        st.subheader("💊 Remedies")
        treatment_steps = remedies.get(predicted_class, ["No remedy available."])
        for step in treatment_steps:
            st.markdown(f"- {step}")