import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess_clip = clip.load("ViT-B/32", device=device)

larva_texts = [
    "a mosquito larva",
    "an Aedes mosquito larva",
    "an Anopheles mosquito larva",
    "a Culex mosquito larva",
]
text_tokens = clip.tokenize(larva_texts).to(device)

def is_larva_image(image_pil, threshold=0.28):
    image_input = preprocess_clip(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_tokens)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).squeeze(0)
        max_sim = similarity.max().item()

    return max_sim >= threshold, max_sim
