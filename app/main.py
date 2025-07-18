import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import clip
from torchvision.models import efficientnet_b0, mobilenet_v3_small, resnet50
from timm import create_model
import time
import sys
import os
import io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from clip_filter import is_larva_image

# Set up Streamlit
st.set_page_config(page_title="🦟 Larvae Classifier", layout="centered")
st.title("🦟 Mosquito Larvae Classifier")
st.caption("Choose model, upload image or use webcam → Get prediction or 'No larvae detected'!")

# Dark mode toggle
dark_mode = st.toggle("🌙 Dark Mode")
if dark_mode:
    st.markdown("""
        <style>
            body {
                background-color: #111111;
                color: #EEEEEE;
            }
            .stButton>button {
                background-color: #444444;
                color: white;
            }
        </style>
    """, unsafe_allow_html=True)

# Class labels
class_names = ['Aedes', 'Anopheles', 'Culex']
clip_class_names = [
    "a high-quality photo of Aedes mosquito larva in water",
    "a high-quality photo of Anopheles mosquito larva",
    "a high-quality photo of Culex mosquito larva close-up",
    "a random object that is not a mosquito larva"
]

# Load CLIP
@st.cache_resource
def load_clip():
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    return model, preprocess

clip_model, clip_preprocess = load_clip()

# Load classifier models
@st.cache_resource
def load_model(name):
    if name == "EfficientNetB0":
        model = torch.load("model/efficientnetb0_final.pt", map_location="cpu", weights_only=False)
    elif name == "MobileNetV3":
        model = torch.load("model/mobilenetv3_large_final.pt", map_location="cpu", weights_only=False)
    elif name == "ResNet50":
        model = torch.load("model/resnet50_final.pt", map_location="cpu", weights_only=False)
    elif name == "VisionTransformer":
        import timm
        from timm.models.vision_transformer import VisionTransformer
        from timm.layers.patch_embed import PatchEmbed
        torch.serialization.add_safe_globals([VisionTransformer, PatchEmbed, torch.nn.Conv2d])

        # Download full model from Hugging Face (use raw/resolve link)
        import tempfile
        import urllib.request

        vit_url = "https://huggingface.co/FuadKhan2/ViT_model/resolve/main/vit_final.pt"
        with tempfile.NamedTemporaryFile() as tmp:
            urllib.request.urlretrieve(vit_url, tmp.name)
            model = torch.load(tmp.name, map_location="cpu")
            model.eval()
    elif name == "YOLOv8":
        from ultralytics import YOLO
        model = YOLO("model/yolov8m.pt")
    else:
        raise ValueError("Model not recognized")

    if name != "YOLOv8":
        model.eval()
    return model

# Image preprocessing
torch_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Select model ---
model_name = st.selectbox("Select a model", [
    "EfficientNetB0",
    "MobileNetV3",
    "ResNet50",
    "VisionTransformer",
    "YOLOv8"
])
model = load_model(model_name)

# --- Image source toggle ---
st.markdown("### Choose input method:")
input_method = st.radio("Select image input type:", ["Upload Image", "Use Webcam"])

image = None
uploaded_file = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload an image (jpg, png)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

elif input_method == "Use Webcam":
    webcam_img = st.camera_input("📸 Take a photo")
    if webcam_img is not None:
        image = Image.open(webcam_img).convert("RGB")
        uploaded_file = io.BytesIO(webcam_img.getvalue())
        uploaded_file.name = "webcam_capture.jpg"

# --- Process if image exists ---
if image:
    st.image(image, caption="Input Image", use_container_width=True)

    with st.spinner("Analyzing..."):

        st.write("✅ Image opened successfully")

        # --- CLIP Similarity Filtering ---
        st.write("📊 Running CLIP similarity check...")
        threshold = st.slider("CLIP similarity threshold", 0.1, 0.5, 0.28, 0.01)
        is_larva, sim_score = is_larva_image(image, threshold=threshold)

        st.write(f"🔎 Similarity score with larva descriptions: `{sim_score:.2f}`")

        if not is_larva:
            st.error("🛑 Image does not resemble a mosquito larva. Prediction aborted.")
            st.stop()
        else:
            st.success("✅ CLIP similarity check passed — looks like a larva.")

        # --- CLIP class-name filter ---
        # st.write("🔍 Running CLIP class-name filter...")
        # image_clip = clip_preprocess(image).unsqueeze(0)
        # with torch.no_grad():
        #     text = clip.tokenize(clip_class_names)
        #     clip_logits = clip_model(image_clip, text)[0]
        #     clip_probs = clip_logits.softmax(dim=-1).squeeze()
        #     best_idx = clip_probs.argmax().item()
        #     clip_label = clip_class_names[best_idx]
        #     confidence = clip_probs[best_idx].item()

        # st.write(f"🔎 CLIP class-name result: {clip_label} ({confidence:.2f})")

        # st.write("🔬 CLIP Class Probabilities:")
        # for i, name in enumerate(clip_class_names):
        #     st.write(f"- {name}: {clip_probs[i].item():.2f}")

        # --- Inference ---
        st.write("⚙️ Running model inference...")

        if model_name != "YOLOv8":
            input_tensor = torch_transform(image).unsqueeze(0)
            with torch.no_grad():
                start_time = time.time()
                outputs = model(input_tensor)
                inference_time = time.time() - start_time
                pred_class = outputs.argmax(1).item()
                prob = torch.softmax(outputs, dim=1)[0][pred_class].item()

            st.success(f"✅ Prediction: **{class_names[pred_class]}** ({prob * 100:.2f}%)")
            st.write(f"🕒 Inference Time: {inference_time:.2f} seconds")

            result_text = f"Image: {uploaded_file.name}\nPrediction: {class_names[pred_class]}\nConfidence: {prob:.4f}"
            st.download_button("📥 Download Prediction", result_text, file_name="prediction.txt")

        else:
            results = model(image)
            boxes = results[0].boxes
            if boxes:
                st.success(f"✅ **YOLO Prediction:** Larvae detected!")
                st.image(results[0].plot(), caption="Detected larvae", use_container_width=True)
            else:
                st.warning("⚠️ No larvae detected.")

# Footer
st.markdown("---")
st.caption("Made with ❤️ using PyTorch + Streamlit + OpenAI CLIP")
