import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os
import requests

# Set the page configuration
st.set_page_config(page_title="MVTec Defect Classifier", layout="wide")

# ---
# This section defines the functions first
# ---

def download_file(url, filename):
    """Downloads a file from a URL, showing a progress spinner in Streamlit."""
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

@st.cache_resource
def load_model(num_classes_param):
    """Loads the ResNet50 model, downloading it if necessary."""
    model_path = "resnet50_mvtec_20epochs.pth"
    # IMPORTANT: This is the correct direct download link for your model
    model_url = "https://huggingface.co/Baji123/resnet50_mvtec/resolve/main/resnet50_mvtec_20epochs.pth"
    
    download_file(model_url, model_path)
    
    st.info("✅ Model file is ready. Loading into memory...")
    
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes_param)
    
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    st.success("🎉 Model loaded successfully! The app is ready.")
    return model

# ---
# This section runs the main part of the script
# ---

# Step 1: Load class names from JSON
classes_file = "classes.json"
if not os.path.exists(classes_file):
    st.error(f"{classes_file} not found! Make sure it is in your GitHub repository.")
    st.stop()

with open(classes_file, "r") as f:
    classes = json.load(f)

# Step 2: Define num_classes **before** using it
num_classes = len(classes)

# Step 3: Now, call load_model and pass num_classes to it
model = load_model(num_classes)

# Step 4: Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---
# Streamlit User Interface
# ---
st.title("🧠 MVTec Defect Classifier")
st.write("Upload an image of an object to predict its defect type.")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file:
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
