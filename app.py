"""Image classifier demo: draw or upload a digit (NumPy CNN), or upload a satellite image (ResNet-18 if trained).

Run:  python -m vision.evaluate  (once)   then   streamlit run app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from vision.evaluate import load_data, saliency  # noqa: E402
from vision.numpy_cnn import CNN  # noqa: E402

st.set_page_config(page_title="Image Classifier", page_icon="🖼️", layout="wide")


@st.cache_resource
def digit_model() -> CNN:
    path = ROOT / "models" / "numpy_cnn_digits.npz"
    if not path.exists():
        from vision.evaluate import main
        main()
    m = CNN()
    m.load_state_dict(dict(np.load(path)))
    return m


def to_digit_tensor(img: Image.Image) -> np.ndarray:
    """Any image → 8×8 greyscale in [0,1], dark ink on light background inverted to match training data."""
    g = np.asarray(img.convert("L").resize((8, 8), Image.Resampling.LANCZOS), dtype=float) / 255.0
    if g.mean() > 0.5:  # white background → invert so strokes are bright like the training digits
        g = 1 - g
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    return g[None, None]


model = digit_model()
metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
st.title("🖼️ Image Classification")
st.caption(f"CNN implemented from scratch in NumPy · {metrics['cnn']['parameters']:,} parameters · "
           f"test accuracy {metrics['results']['NumPy CNN + augmentation']['accuracy']:.1%}")

tab1, tab2, tab3 = st.tabs(["✍️ Digits (NumPy CNN)", "🛰️ Satellite images (ResNet-18)", "📊 Model comparison"])

with tab1:
    left, right = st.columns(2)
    with left:
        source = st.radio("Input", ["Draw", "Pick a test image", "Upload"], horizontal=True)
        x = None
        if source == "Draw":
            try:
                from streamlit_drawable_canvas import st_canvas
                canvas = st_canvas(stroke_width=22, stroke_color="#000000", background_color="#FFFFFF", height=240, width=240,
                                   drawing_mode="freedraw", key="canvas")
                if canvas.image_data is not None and canvas.image_data[..., :3].min() < 250:
                    x = to_digit_tensor(Image.fromarray(canvas.image_data.astype("uint8")))
            except ImportError:
                st.info("Install `streamlit-drawable-canvas` to draw digits; meanwhile pick a test image.")
        elif source == "Pick a test image":
            _, _, _, _, x_te, y_te = load_data()
            i = st.slider("Test image", 0, len(x_te) - 1, 7)
            x = x_te[i:i + 1]
            st.caption(f"True label: {y_te[i]}")
        else:
            f = st.file_uploader("Image of a single digit", type=["png", "jpg", "jpeg"])
            if f:
                x = to_digit_tensor(Image.open(f))
    with right:
        if x is not None:
            p = model.predict_proba(x)[0]
            st.markdown(f"## Prediction: **{p.argmax()}** ({p.max():.0%})")
            st.bar_chart(pd.Series(p, index=[str(d) for d in range(10)], name="probability"))
            c1, c2 = st.columns(2)
            c1.image(np.kron(x[0, 0], np.ones((24, 24))), caption="What the model sees (8×8)", clamp=True)
            sal = saliency(model, x[0], int(p.argmax()))
            c2.image(np.kron(sal / (sal.max() + 1e-8), np.ones((24, 24))), caption="Saliency: pixels driving the decision", clamp=True)

with tab2:
    weights = ROOT / "models" / "resnet18_eurosat.pt"
    if not weights.exists():
        st.info("Train the ResNet-18 on EuroSAT first (about 10 minutes on a free Colab GPU): "
                "`notebooks/transfer_learning_colab.ipynb`, then copy `models/resnet18_eurosat.pt` here.")
    else:
        import torch
        from vision.transfer_learning import build_model, build_transforms, grad_cam
        ckpt = torch.load(weights, map_location="cpu")
        net = build_model(len(ckpt["classes"]))
        net.load_state_dict(ckpt["state_dict"])
        f = st.file_uploader("Satellite image tile", type=["png", "jpg", "jpeg", "tif"], key="sat")
        if f:
            img = Image.open(f).convert("RGB")
            t = build_transforms()[1](img)
            with torch.no_grad():
                probs = torch.softmax(net.eval()(t[None]), 1)[0].numpy()
            st.markdown(f"## {ckpt['classes'][probs.argmax()]} ({probs.max():.0%})")
            st.bar_chart(pd.Series(probs, index=ckpt["classes"]).sort_values(ascending=False))
            cam = grad_cam(net, t)
            a, b = st.columns(2)
            a.image(img, caption="input", width=250)
            b.image(np.kron(cam, np.ones((32, 32))), caption="Grad-CAM: where the network looked", clamp=True, width=250)

with tab3:
    st.dataframe(pd.DataFrame(metrics["results"]).T, use_container_width=True)
    for fig in ["learning_curves.png", "confusion_matrix.png", "errors_saliency.png", "conv_filters.png"]:
        st.image(str(ROOT / "reports" / "figures" / fig))
