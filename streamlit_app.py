import os, io
from typing import Tuple
import numpy as np
import torch, lpips
from PIL import Image
import streamlit as st

# Safer cache location for model weights on hosted runners
os.environ.setdefault("XDG_CACHE_HOME", ".cache")
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

@st.cache_resource
def load_lpips(net: str = "alex"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = lpips.LPIPS(net=net).to(device).eval()
    return model, device

def to_tensor(img: Image.Image, size: Tuple[int, int] = (256, 256), device: str = "cpu"):
    img = img.convert("RGB").resize(size, Image.LANCZOS)
    arr = (np.asarray(img).astype("float32")/255.0)*2.0 - 1.0
    return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(device)

st.set_page_config(page_title="LPIPS Duplicate Checker", page_icon="🖼️", layout="centered")
st.title("🖼️ LPIPS Duplicate Checker")

col = st.columns(2)
with col[0]:
    f1 = st.file_uploader("Image 1", type=["png","jpg","jpeg","webp"], key="img1")
with col[1]:
    f2 = st.file_uploader("Image 2", type=["png","jpg","jpeg","webp"], key="img2")

net = st.selectbox("Backbone", ["alex","vgg"], index=0)
model, device = load_lpips(net)

THRESHOLDS = {
    "alex": (0.06, 0.08),
    "vgg":  (0.10, 0.12),
}

if f1 and f2:
    img1 = Image.open(io.BytesIO(f1.read()))
    img2 = Image.open(io.BytesIO(f2.read()))
    st.image([img1, img2], caption=["Image 1","Image 2"], width=256)

    if st.button("Compare"):
        a = to_tensor(img1, device=device)
        b = to_tensor(img2, device=device)
        with torch.no_grad():
            d = float(model(a, b).item())

        tight, balanced = THRESHOLDS[net]
        verdict = "Duplicate" if d <= tight else "Near-duplicate" if d <= balanced else "Different"

        st.metric("LPIPS", f"{d:.4f}", help="Lower = more similar")
        st.success(f"Verdict: **{verdict}**  •  (tight≤{tight}, balanced≤{balanced})")

st.caption("Tip: Calibrate thresholds on your own data.")
