import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os

st.set_page_config(page_title="MVTec Defect Classifier", layout="wide")

# -------------------------------
# Load class names
# -------------------------------
classes_file = "classes.json"
if not os.path.exists(classes_file):
    st.error(f"{classes_file} not found! Make sure it is in the same folder as this script.")
    st.stop()

with open(classes_file, "r") as f:
    classes = json.load(f)

num_classes = len(classes)

# -------------------------------
# Load the trained model
# -------------------------------
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model_path = r"resnet50_mvtec_20epochs.pth"  # your trained model
    if not os.path.exists(model_path):
        st.error(f"{model_path} not found! Place the trained model in the same folder.")
        st.stop()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -------------------------------
# Image transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------------
# Streamlit UI
# -------------------------------
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

        # Optional: show confidence score
        probs = torch.softmax(outputs, dim=1)
        confidence = probs[0][pred_idx].item()
        st.write(f"Confidence: {confidence*100:.2f}%")

    except Exception as e:
        st.error(f"Error processing the image: {e}")
