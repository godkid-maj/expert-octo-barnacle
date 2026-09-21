"""
app.py
------
Streamlit front end for the fine-tuned image classifier.

Run with:
    streamlit run app.py

Expects model/model.pth and model/classes.json to exist (produced by
train.py). Falls back to plain ImageNet-pretrained ResNet18 predictions
if no fine-tuned model is found yet, so the app is demoable immediately.
"""

import json
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "model.pth"
CLASSES_PATH = MODEL_DIR / "classes.json"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

st.set_page_config(page_title="Image Classifier", page_icon="🖼️", layout="centered")


@st.cache_resource
def load_model():
    """Loads the fine-tuned model if available, otherwise a generic
    ImageNet-pretrained ResNet18 so the app still works out of the box."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if MODEL_PATH.exists() and CLASSES_PATH.exists():
        with open(CLASSES_PATH) as f:
            class_names = json.load(f)
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        source = "fine-tuned model"
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        class_names = models.ResNet18_Weights.IMAGENET1K_V1.meta["categories"]
        source = "generic ImageNet weights (no fine-tuned model found yet — run train.py)"

    model.eval().to(device)
    return model, class_names, device, source


def preprocess(image: Image.Image, img_size: int = 224) -> torch.Tensor:
    tfms = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tfms(image.convert("RGB")).unsqueeze(0)


def predict(model, device, tensor, class_names, top_k=5):
    with torch.no_grad():
        logits = model(tensor.to(device))
        probs = F.softmax(logits, dim=1)[0]
    top_probs, top_idxs = probs.topk(min(top_k, len(class_names)))
    return [(class_names[i], float(p)) for p, i in zip(top_probs, top_idxs)]


def main():
    st.title("🖼️ Image Classifier")
    st.write("Upload a photo and the model will predict what's in it.")

    model, class_names, device, source = load_model()
    st.caption(f"Model: {source} · {len(class_names)} classes")

    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded is not None:
        image = Image.open(uploaded)
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Classifying..."):
            tensor = preprocess(image)
            results = predict(model, device, tensor, class_names, top_k=5)

        with col2:
            st.subheader("Predictions")
            top_label, top_conf = results[0]
            st.metric("Top prediction", top_label.replace("_", " "), f"{top_conf:.1%} confidence")
            st.write("")
            for label, conf in results:
                st.write(label.replace("_", " "))
                st.progress(conf)
    else:
        st.info("👆 Upload a JPG or PNG to get started.")


if __name__ == "__main__":
    main()
