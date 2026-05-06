import streamlit as st
import torch
import os
from model import Generator
from utils import generate_images

# Page config
st.set_page_config(page_title="GAN Generator", layout="wide")

st.title("🎨 GAN Image Generator")
st.markdown("Compare **DCGAN vs WGAN-GP** outputs")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.write(f"Device: {device}")

# Paths
DCGAN_PATH = "G_dcgan_final.pth"
WGAN_PATH = "G_wgan_final.pth"

# 🔍 Check files exist
if not os.path.exists(DCGAN_PATH):
    st.error("❌ G_dcgan_final.pth NOT found!")
    st.stop()

if not os.path.exists(WGAN_PATH):
    st.error("❌ G_wgan_final.pth NOT found!")
    st.stop()

# ✅ Load models safely
@st.cache_resource
def load_models():
    z_dim = 100

    # DCGAN
    G_dcgan = Generator(z_dim).to(device)
    state_dcgan = torch.load(DCGAN_PATH, map_location=device)
    G_dcgan.load_state_dict(state_dcgan)
    G_dcgan.eval()

    # WGAN
    G_wgan = Generator(z_dim).to(device)
    state_wgan = torch.load(WGAN_PATH, map_location=device)
    G_wgan.load_state_dict(state_wgan)
    G_wgan.eval()

    return G_dcgan, G_wgan

G_dcgan, G_wgan = load_models()

# Sidebar controls
st.sidebar.header("Settings")
model_choice = st.sidebar.radio("Choose Model", ["DCGAN", "WGAN-GP"])
num_images = st.sidebar.slider("Number of Images", 4, 64, 16, step=4)

# Generate
if st.button("🚀 Generate Images"):
    if model_choice == "DCGAN":
        img = generate_images(G_dcgan, device=device, n=num_images)
    else:
        img = generate_images(G_wgan, device=device, n=num_images)

    st.image(img, caption=f"{model_choice} Output", use_container_width=True)

# Compare section
st.markdown("---")
st.subheader("🔍 Compare Both Models")

if st.button("Compare Models"):
    col1, col2 = st.columns(2)

    img1 = generate_images(G_dcgan, device=device)
    img2 = generate_images(G_wgan, device=device)

    with col1:
        st.image(img1, caption="DCGAN", use_container_width=True)

    with col2:
        st.image(img2, caption="WGAN-GP", use_container_width=True)
