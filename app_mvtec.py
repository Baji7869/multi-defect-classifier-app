import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os
import requests

st.set_page_config(page_title="MVTec Defect Classifier", layout="wide")

# -------------------------------
# Load class names
# -------------------------------
classes_file = "classes.json"
if not os.path.exists(classes_file):
    st.error(f"{classes_file} not found! Make sure it is in your GitHub repository.")
    st.stop()

with open(classes_file, "r") as f:
    classes = json.load(f)

num_classes = len(classes)

# -------------------------------
# Function to download the model
# -------------------------------
def download_file(url, filename):
    if not os.path.exists(filename):
        try:
            with st.spinner(f"Downloading model... (This may take a minute on the first run)"):
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(filename, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
        except requests.exceptions.RequestException as e:
            st.error(f"Error downloading model: {e}")
            st.stop()

# -------------------------------
# Load the trained model
# -------------------------------
@st.cache_resource
# FIX 1: Add 'num_classes' as a parameter to the function
def load_model(num_classes_param):
    model_path = "resnet50_mvtec_20epochs.pth"
    model_url = "https://huggingface.co/Baji123/resnet50_mvtec/resolve/main/resnet50_mvtec_20epochs.pth"
    
    download_file(model_url, model_path)
    
    st.info("✅ Model file is ready. Loading into memory...")
    
    model = models.resnet50(weights=None)
    # Use the parameter passed to the function
    model.fc = nn.Linear(model.fc.in_features, num_classes_param)
    
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    st.success("🎉 Model loaded! The app is ready.")
    return model

# FIX 2: Pass 'num_classes' when you call the function
model = load_model(num_classes)

# -------------------------------
# Image transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🧠 MVTec Defect Classifier")
st.write("Upload an image of an object to predict its defect type.")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # (The rest of your UI code stays exactly the same)
    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)
        input_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, pred = outputs.max(1)
        pred_idx = pred.item()
        if pred_idx >= len(classes):
            class_name = f"Class ID {pred_idx} (Unknown)"
            status = "Unknown"
        else:
            class_name = classes[pred_idx]
            status = "Good (No Defect)" if "good" in class_name.lower() else "Defected"
        st.success(f"Predicted Class: {class_name}")
        st.info(f"Status: {status}")
        probs = torch.softmax(outputs, dim=1)
        confidence = probs[0][pred_idx].item()
        st.write(f"Confidence: {confidence*100:.2f}%")
    except Exception as e:
        st.error(f"Error processing the image: {e}")
