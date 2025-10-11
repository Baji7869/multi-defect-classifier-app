import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os
import requests
from tqdm import tqdm

# ... (keep your st.set_page_config and class loading code) ...

# -------------------------------\n# NEW - Function to download the model\n# -------------------------------
def download_file(url, filename):
    """Downloads a file from a URL, showing a progress bar in Streamlit."""
    if os.path.exists(filename):
        # If file is already there, no need to download again.
        return

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size_in_bytes = int(r.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            
            progress_bar = st.progress(0, text=f"Downloading model ({total_size_in_bytes/1e6:.1f} MB)...")
            
            with open(filename, 'wb') as f:
                downloaded_size = 0
                for chunk in r.iter_content(chunk_size=block_size):
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    progress = int(100 * downloaded_size / total_size_in_bytes)
                    progress_bar.progress(progress, text=f"Downloading model... {progress}%")
            
            progress_bar.empty() # Remove the progress bar after completion
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading model: {e}")
        st.stop()


# -------------------------------\n# Load the trained model\n# -------------------------------

@st.cache_resource
def load_model():
    model_path = "resnet50_mvtec_20epochs.pth"
    model_url = "https://huggingface.co/Baji123/resnet50_mvtec/resolve/main/resnet50_mvtec_20epochs.pth"

    # Add a message before the download
    st.info("🚀 First-time setup: Downloading the model... This may take a few minutes.")

    download_file(model_url, model_path)

    # Add a message after the download
    st.info("✅ Download complete. Loading model into memory...")

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    # Add a final success message
    st.success("🎉 Model loaded successfully! The app is ready.")
    
    return model
