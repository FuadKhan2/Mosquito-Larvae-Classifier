# 🦟 Mosquito Larvae Classifier (Mobile + Web App)

This is a responsive, mobile-friendly Streamlit app that classifies mosquito larvae using deep learning models. Users can either upload an image or capture one in real time via webcam or mobile camera.

> Developed by **MD. Fuad Khan** and **Md. Bodrul Islam** as part of a final-year CSE thesis project.

## Features

- Supports 5 trained models:
  - **EfficientNetB0**
  - **MobileNetV3**
  - **ResNet50**
  - **Vision Transformer (ViT)**
  - **YOLOv8** (for detection)

- 📷 Choose input mode:
  - Upload from file
  - Capture using webcam / mobile camera

## Dataset Info

This project uses a custom image dataset of mosquito larvae collected and labeled for classification and detection tasks.Maximum of the images are taken using Mobile Phone by me and my thesis partner Md. Bodrul Islam and some of the images are collected from online sources.

- **Total Images:** 1,890
- **Classes:** Aedes, Anopheles, Culex
- **Folder Split:**
  - **Train:**
    - Aedes: 732
    - Anopheles: 246
    - Culex: 345
  - **Validation:**
    - Aedes: 156
    - Anopheles: 50
    - Culex: 75
  - **Test:**
    - Aedes: 157
    - Anopheles: 53
    - Culex: 76

- **Total per Class:**
  - **Aedes:** 1,045 images
  - **Anopheles:** 349 images
  - **Culex:** 496 images

- **Dataset Creator:** [MD. Fuad Khan and Md. Bodrul Islam](https://www.kaggle.com/mdfuadkhan)

- **Kaggle Dataset Link:**  
  [Mosquito Larvae Image Dataset (by MD. Fuad Khan and Md. Bodrul Islam)](https://www.kaggle.com/datasets/mdfuadkhan/mosquito-larvae-image-dataset/data)

> ⚠️ Please cite or credit the creator if you use the dataset in other projects or papers.




## Model Info

- **Classification Models** (EfficientNet, ResNet, MobileNet, ViT):
  - Trained on a labeled dataset of mosquito larvae (Aedes, Culex, Anopheles)
  - Output: predicted class + confidence score

- **YOLOv8 Detector**:
  - Detects larvae from raw images
  - Outputs bounding boxes and labels

## Technologies Used

- Python
- Streamlit
- PyTorch / TorchVision / timm
- YOLOv8 (Ultralytics)
- OpenCV
- scikit-learn
- Matplotlib, Seaborn

## How to Run Locally

```bash
git clone https://github.com/yourusername/larvae-classifier-app.git
cd larvae-classifier-app
pip install -r requirements.txt
streamlit run main.py
