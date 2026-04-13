import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import json
import os
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
    # This is the correct direct download link for your model
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
# ── NEW: U-Net Architecture ──────────────────────────────────────────────────
# ---

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)


class UNet(nn.Module):
    """U-Net for pixel-level defect segmentation. Output: sigmoid probability map."""
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        self.pool  = nn.MaxPool2d(2, 2)
        ch = in_channels
        for f in features:
            self.downs.append(ConvBlock(ch, f)); ch = f
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.ups.append(ConvBlock(f * 2, f))
        self.final = nn.Conv2d(features[0], 1, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[-(i // 2 + 1)]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)
        return torch.sigmoid(self.final(x))


@st.cache_resource
def load_unet():
    """
    Loads U-Net weights from HuggingFace.
    Upload your trained unet_mvtec.pth to:
      https://huggingface.co/Baji123/resnet50_mvtec/resolve/main/unet_mvtec.pth
    and this will auto-download it.
    Returns (model, True) if loaded, (None, False) if not available.
    """
    unet_path = "unet_mvtec.pth"
    unet_url  = "https://huggingface.co/Baji123/resnet50_mvtec/resolve/main/unet_mvtec.pth"

    if not os.path.exists(unet_path):
        try:
            with st.spinner("Downloading U-Net segmentation model..."):
                r = requests.get(unet_url, stream=True, timeout=15)
                r.raise_for_status()
                with open(unet_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except Exception:
            return None, False

    try:
        unet = UNet()
        unet.load_state_dict(torch.load(unet_path, map_location="cpu"))
        unet.eval()
        return unet, True
    except Exception:
        return None, False

# ---
# ── NEW: Damage Estimation & Decision Logic ──────────────────────────────────
# ---

ACCEPT_THRESHOLD = 2.0    # damage% < 2   → ACCEPT
REWORK_THRESHOLD = 10.0   # damage% < 10  → REWORK, else → REJECT

def estimate_damage(pred_mask_tensor, threshold=0.5):
    """
    Computes damage percentage from U-Net predicted sigmoid mask.
    Returns: (damage_pct float, binary_mask np.ndarray H×W)
    """
    mask   = pred_mask_tensor.squeeze()
    binary = (mask > threshold).float()
    pct    = (binary.sum() / binary.numel()).item() * 100.0
    return round(pct, 2), binary.numpy()


def make_decision(damage_pct):
    """Returns (decision str, note str, color str)."""
    if damage_pct < ACCEPT_THRESHOLD:
        return "ACCEPT", f"Damage {damage_pct:.2f}% < {ACCEPT_THRESHOLD}% threshold — product OK", "#1a7a4a"
    elif damage_pct < REWORK_THRESHOLD:
        return "REWORK", f"Damage {damage_pct:.2f}% in [{ACCEPT_THRESHOLD}%–{REWORK_THRESHOLD}%) — needs rework", "#b85c00"
    else:
        return "REJECT", f"Damage {damage_pct:.2f}% ≥ {REWORK_THRESHOLD}% threshold — reject product", "#a01a1a"


def run_segmentation(unet, img_pil, threshold=0.5):
    """
    Runs U-Net on a PIL image.
    Returns: (pred_mask_tensor, damage_pct, binary_mask_np, overlay_np)
    """
    seg_t = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_np = np.array(img_pil.resize((256, 256)))
    img_tensor = seg_t(img_pil).unsqueeze(0)

    with torch.no_grad():
        pred_mask = unet(img_tensor).squeeze(0).cpu()

    damage_pct, binary_mask = estimate_damage(pred_mask, threshold)

    # Build red overlay
    overlay = img_np.copy()
    mask_rgb = np.zeros_like(overlay)
    mask_rgb[:, :, 0] = (binary_mask * 255).astype(np.uint8)
    overlay = cv2.addWeighted(overlay, 0.55, mask_rgb, 0.45, 0)

    # Build heatmap (colormap on raw prob)
    prob_map = pred_mask.squeeze().numpy()
    heatmap  = cv2.applyColorMap(np.uint8(255 * prob_map), cv2.COLORMAP_JET)
    heatmap  = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return pred_mask, damage_pct, binary_mask, overlay, heatmap

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

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output)

        self.model.zero_grad()
        output[0, class_idx].backward()

        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam

# Step 3: Now, call load_model and pass num_classes to it
model = load_model(num_classes)

# Step 4: Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def generate_gradcam(model, img):
    img_np = np.array(img.resize((224, 224)))

    pil_img = Image.fromarray(img_np)
    tensor_img = transform(pil_img).unsqueeze(0)

    gradcam = GradCAM(model, model.layer4[-1])
    cam = gradcam.generate(tensor_img)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = heatmap * 0.5 + img_np * 0.5

    return img_np, overlay.astype(np.uint8)

# ── NEW: Load U-Net (non-blocking — app still works without it) ──────────────
unet_model, unet_available = load_unet()

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

        # ✅ Grad-CAM visualization
        st.subheader("🔥 Grad-CAM Visualization")

        original, cam_image = generate_gradcam(model, img)

        col1, col2 = st.columns(2)

        with col1:
            st.image(original, caption="Original", use_container_width=True)

        with col2:
            st.image(cam_image, caption="Grad-CAM Heatmap", use_container_width=True)

    except Exception as e:
        st.error(f"Error processing the image: {e}")

    # ── NEW SECTION: Segmentation + Damage Estimation + Decision ────────────────
    st.divider()
    st.subheader("🎭 Pixel-Level Defect Segmentation")

    if not unet_available:
        st.warning(
            "⚠️ U-Net segmentation model not found.\n\n"
            "**To enable this section:**\n"
            "1. Train U-Net using the Kaggle notebook (`train_unet()` function).\n"
            "2. Upload `unet_mvtec.pth` to your HuggingFace repo: "
            "`https://huggingface.co/Baji123/resnet50_mvtec/`\n"
            "3. The app will auto-download it on next run."
        )
    else:
        try:
            # ── Threshold slider ──
            seg_threshold = st.slider(
                "Segmentation Threshold",
                min_value=0.1, max_value=0.9, value=0.5, step=0.05,
                help="Lower → detect more pixels as defective. Higher → stricter detection."
            )

            with st.spinner("Running U-Net segmentation..."):
                pred_mask, damage_pct, binary_mask, overlay_np, heatmap_np = run_segmentation(
                    unet_model, img, threshold=seg_threshold
                )

            # ── Show segmentation results ──
            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(
                    np.array(img.resize((256, 256))),
                    caption="Original Image",
                    use_container_width=True
                )
            with col2:
                # Binary mask as red-highlighted image
                mask_display = np.zeros((256, 256, 3), dtype=np.uint8)
                mask_display[:, :, 0] = (binary_mask * 255).astype(np.uint8)
                st.image(mask_display, caption="Defect Mask (Red = Defective)", use_container_width=True)

            with col3:
                st.image(overlay_np, caption="Damage Overlay", use_container_width=True)

            # ── Heatmap row ──
            st.image(heatmap_np, caption="Defect Probability Heatmap (Blue→Red = Low→High)", use_container_width=True)

            # ── Damage Estimation ────────────────────────────────────────────────
            st.divider()
            st.subheader("📊 Damage Estimation")

            col_dmg1, col_dmg2, col_dmg3 = st.columns(3)

            total_pixels   = binary_mask.size
            defect_pixels  = int(binary_mask.sum())
            healthy_pixels = total_pixels - defect_pixels

            with col_dmg1:
                st.metric("🔴 Defect Pixels",  f"{defect_pixels:,}")
            with col_dmg2:
                st.metric("🟢 Healthy Pixels", f"{healthy_pixels:,}")
            with col_dmg3:
                st.metric("⚠️ Damage Percentage", f"{damage_pct:.2f}%")

            # Progress bar for damage
            st.write("**Damage Level:**")
            st.progress(min(damage_pct / 100.0, 1.0))

            # Pixel distribution pie chart
            fig, ax = plt.subplots(figsize=(5, 3), facecolor="none")
            ax.pie(
                [healthy_pixels, defect_pixels],
                labels=["Healthy", "Defective"],
                colors=["#2ecc71", "#e74c3c"],
                autopct="%1.1f%%",
                startangle=90,
                textprops={"color": "white", "fontsize": 11}
            )
            ax.set_facecolor("none")
            fig.patch.set_alpha(0)
            st.pyplot(fig)
            plt.close(fig)

            # ── Decision Module ──────────────────────────────────────────────────
            st.divider()
            st.subheader("✅ Quality Decision")

            # Threshold customization
            with st.expander("⚙️ Decision Thresholds (click to customize)"):
                accept_thr = st.number_input(
                    "Accept threshold (%)", value=ACCEPT_THRESHOLD,
                    min_value=0.0, max_value=100.0, step=0.5,
                    help="Damage below this → ACCEPT"
                )
                rework_thr = st.number_input(
                    "Rework threshold (%)", value=REWORK_THRESHOLD,
                    min_value=0.0, max_value=100.0, step=0.5,
                    help="Damage below this (but above accept) → REWORK, else → REJECT"
                )

            # Compute decision using custom thresholds
            if damage_pct < accept_thr:
                decision = "ACCEPT"
                note  = f"Damage {damage_pct:.2f}% < {accept_thr}% — product OK"
                color = "#1a7a4a"
            elif damage_pct < rework_thr:
                decision = "REWORK"
                note  = f"Damage {damage_pct:.2f}% in [{accept_thr}%–{rework_thr}%) — needs rework"
                color = "#b85c00"
            else:
                decision = "REJECT"
                note  = f"Damage {damage_pct:.2f}% ≥ {rework_thr}% — reject product"
                color = "#a01a1a"

            DECISION_ICON = {"ACCEPT": "✅", "REWORK": "⚠️", "REJECT": "❌"}
            DECISION_BG   = {"ACCEPT": "#0d2e1a", "REWORK": "#2e1a00", "REJECT": "#2e0000"}

            st.markdown(
                f"""
                <div style="
                    background-color: {DECISION_BG[decision]};
                    border: 2px solid {color};
                    border-radius: 12px;
                    padding: 24px 32px;
                    text-align: center;
                    margin: 16px 0;
                ">
                    <div style="font-size: 3rem; margin-bottom: 8px;">{DECISION_ICON[decision]}</div>
                    <div style="font-size: 2rem; font-weight: 800; color: {color}; letter-spacing: 0.1em;">
                        {decision}
                    </div>
                    <div style="font-size: 1rem; color: #cccccc; margin-top: 10px;">{note}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ── Full Pipeline Summary Table ──────────────────────────────────────
            st.divider()
            st.subheader("📋 Full Pipeline Summary")

            summary_data = {
                "Module": ["Classification (ResNet50)", "Grad-CAM", "Segmentation (U-Net)", "Damage Estimation", "Quality Decision"],
                "Result": [
                    f"{class_name} ({confidence*100:.1f}% confidence)",
                    "Defect region highlighted ✓",
                    f"Mask generated at threshold {seg_threshold}",
                    f"{damage_pct:.2f}% of pixels defective",
                    f"{DECISION_ICON[decision]} {decision} — {note}"
                ]
            }

            col_a, col_b = st.columns([1, 2])
            with col_a:
                for m in summary_data["Module"]:
                    st.markdown(f"**{m}**")
            with col_b:
                for r in summary_data["Result"]:
                    st.markdown(r)

        except Exception as e:
            st.error(f"Segmentation error: {e}")
            st.info("Make sure the U-Net model was trained on 256×256 images with 3 input channels.")
