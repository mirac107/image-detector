import io
from typing import Tuple

import torch
import lpips
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
loss = lpips.LPIPS(net="alex").to(device).eval()

app = FastAPI(title="LPIPS micro-service")

def to_tensor(img: Image.Image, size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
    img = img.convert("RGB").resize(size, Image.LANCZOS)
    arr = (np.asarray(img).astype("float32") / 255.0) * 2.0 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

@app.post("/score")
async def score(
    img1: UploadFile = File(..., description="Reference image"),
    img2: UploadFile = File(..., description="Candidate image"),
):
    a = to_tensor(Image.open(io.BytesIO(await img1.read())))
    b = to_tensor(Image.open(io.BytesIO(await img2.read())))
    with torch.no_grad():
        d = float(loss(a, b).item())
    return {"lpips": d}
