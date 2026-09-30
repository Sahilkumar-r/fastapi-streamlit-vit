"""Streamlit frontend: upload an image and see what the ViT thinks it is."""
import os

import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="ViT Image Classifier", page_icon="🖼️", layout="centered")
st.title("🖼️ Vision Transformer Image Classifier")
st.caption("Upload a picture and the ViT will identify what is in it.")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp", "bmp"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Your image", use_container_width=True)

    if st.button("Classify", type="primary"):
        with st.spinner("Analyzing..."):
            try:
                resp = requests.post(
                    f"{API_URL}/predict",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    timeout=60,
                )
                resp.raise_for_status()
                result = resp.json()
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot reach the backend at {API_URL}. Is it running?")
                st.stop()
            except requests.exceptions.HTTPError:
                st.error(resp.json().get("detail", "Request failed."))
                st.stop()

        preds = result["predictions"]
        best = preds[0]
        st.success(f"**{best['label']}** ({best['confidence']:.1%})")

        st.subheader("Top predictions")
        for p in preds:
            st.write(p["label"])
            st.progress(min(max(p["confidence"], 0.0), 1.0), text=f"{p['confidence']:.1%}")
        st.caption(f"Model: {result['model']}")
