# app.py

import streamlit as st
import torch
import clip
from PIL import Image

# PLACE THE DEVICE AND MODEL LOADING HERE
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Set your defect prompts
prompts = ["normal foil", "foil with scratch defect", "foil with hole defect", "foil with edge defect"]
text_tokens = clip.tokenize(prompts).to(device)

# Streamlit UI
st.title("Zero-Shot Defect Detection in Foils using CLIP")

uploaded_file = st.file_uploader("Upload an image of foil", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Get embeddings
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_tokens)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        for i, prompt in enumerate(prompts):
            st.write(f"{prompt}: {similarity[0][i].item() * 100:.2f}%")

        detected_class = prompts[similarity[0].argmax()]
        st.success(f"Predicted: **{detected_class}**")




