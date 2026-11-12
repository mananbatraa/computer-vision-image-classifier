"""Streamlit UI: upload a satellite image and see what land use the model predicts.

    streamlit run app.py
"""
import json
from pathlib import Path

import streamlit as st
from PIL import Image

from classifier.predict import DEFAULT_CHECKPOINT, LOW_CONFIDENCE, Classifier

ROOT = Path(__file__).parent
EXAMPLES = ROOT / "examples"

st.set_page_config(page_title="Land-Use Image Classifier", page_icon="🛰️", layout="centered")
st.title("🛰️ Land-Use Image Classifier")
st.caption("MobileNetV3 fine-tuned on EuroSAT (Sentinel-2 satellite images, 10 land-use classes).")


@st.cache_resource(show_spinner="Loading model…")
def get_classifier():
    return Classifier(DEFAULT_CHECKPOINT)


if not DEFAULT_CHECKPOINT.exists():
    st.error("No trained model found. Run `python -m classifier.train` first.")
    st.stop()
clf = get_classifier()

with st.sidebar:
    st.subheader("Classes")
    st.write("\n".join(f"- {c}" for c in clf.classes))
    metrics_file = ROOT / "reports" / "metrics.json"
    if metrics_file.exists():
        m = json.loads(metrics_file.read_text())
        st.metric("Test accuracy", f"{m['test_accuracy']:.1%}")
        st.metric("Macro F1", f"{m['test_macro_f1']:.3f}")

upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "tif", "tiff"])
examples = sorted(EXAMPLES.glob("*.jpg")) if EXAMPLES.exists() else []
choice = None
if not upload and examples:
    choice = st.selectbox("…or try an example from the test set", examples, format_func=lambda p: p.stem)

source = upload or choice
if source is None:
    st.info("Upload an overhead / satellite image to classify it.")
    st.stop()

image = Image.open(source)
pred = clf.predict(image, top=5)

left, right = st.columns([1, 1.3])
left.image(image, caption="Input", use_container_width=True)
with right:
    st.subheader(pred.label)
    st.write(f"Confidence: **{pred.confidence:.1%}**")
    if pred.uncertain:
        st.warning(f"Low confidence (below {LOW_CONFIDENCE:.0%}). The image may not look like "
                   "the satellite imagery the model was trained on.")
    for label, p in pred.top_k:
        st.progress(p, text=f"{label}: {p:.1%}")
