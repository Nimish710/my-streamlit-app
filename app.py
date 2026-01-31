import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import json
import os

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      

MODEL_PATH = os.path.join(BASE_DIR, "best_model.pth")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")
BREED_INFO_PATH = os.path.join(BASE_DIR, "breed_info.json")


# -----------------------------
# LOAD LABELS
# -----------------------------
with open(LABELS_PATH, "r") as f:
    idx_to_class = json.load(f)

# -----------------------------
# LOAD METADATA
# -----------------------------
with open(BREED_INFO_PATH, "r",encoding='utf-8') as f:
    breed_info = json.load(f)

# -----------------------------
# LOAD MODEL (EfficientNet-B0)
# -----------------------------
def load_model():
    num_classes = len(idx_to_class)

    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(1280, num_classes)

    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


model = load_model()

# -----------------------------
# TRANSFORMS
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# PREDICT FUNCTION
# -----------------------------
def predict(image):
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    breed_name = idx_to_class[str(predicted.item())]
    return breed_name


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🐮 Cow Breed Classification + Market Insights")

uploaded_image = st.file_uploader("Upload a cow image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        breed = predict(image)
        st.success(f"Predicted Breed: **{breed}** 🐄")

        # Show breed metadata
        if breed in breed_info:
            st.subheader("📌 Market Information")
            st.json(breed_info[breed])
        else:

            st.warning("❗ No metadata found for this breed.")  
