import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import json
from ultralytics import YOLO
import os

# Set title
st.title("PanelDetect 🔧💡: Smart Insights for Solar Panel Health")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load class names from JSON
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

#st.sidebar.success("Classes loaded: " + ", ".join(class_names))

# Load the model
@st.cache_resource
def load_model():
    # Define the model architecture (the same as used during training)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # ResNet18 without pre-trained weights
    # Load the trained model weights
    model.fc=torch.nn.Linear(model.fc.in_features,6)
    model.load_state_dict(torch.load("best_resnet18_model.pth", map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

def Modelpreprocess():
    model = load_model()

    # Define image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # Match the normalization used during training
    ])
    return model,transform

# File uploader
def streeamlitfileupload():
    uploaded_file = st.file_uploader("📤 Upload an image of a solar panel", type=["jpg", "jpeg", "png"])
    return uploaded_file

# Image Classifier Prediction
def Imageclassifier(uploaded_file):
    st.subheader("Identifying Solar Panel Conditions")
    model,transform=Modelpreprocess()
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)

        st.write("🔍 Classifying...")
        with torch.no_grad():
            img = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            label = class_names[predicted.item()]

        st.success(f"✅ Prediction: **{label}**")




#Object Detection Prediction
def Objectdetection(uploaded_file):
    st.subheader("Localizing Issues on Solar Panels")
    trained_model_path = 'runs/detect/train2/weights/best.pt'   # or check correct path

    model = YOLO(trained_model_path)

    # Create folder for uploaded images
    uploaded_dir= "uploaded_images"
    os.makedirs(uploaded_dir, exist_ok=True)
    if uploaded_file is not None:
        if uploaded_file is not None:
        # Save uploaded image to folder
            file_path = os.path.join(uploaded_dir, uploaded_file.name)

            # Move file pointer to beginning before reading
            uploaded_file.seek(0)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            #  prediction
            results = model.predict(source=file_path, conf=0.35, save=True)

            #  Locate saved prediction image
            result_folder = results[0].save_dir
            result_img_path = os.path.join(result_folder, uploaded_file.name)

            #  Display result image in Streamlit
            st.image(result_img_path, caption="Detected Output", use_container_width=True)

            # Optional: delete uploaded file
            os.remove(file_path)



def main():
    
    page = st.sidebar.radio("Navigate", ["🔍 Detect Panel Condition", "ℹ️ About / Help"])
    if page == "🔍 Detect Panel Condition":
        uploaded_img_file=streeamlitfileupload()
        if uploaded_img_file is not None:
            #Prediction 1 For imageClassifier
            if st.button('🔬 Identify Panel Coditions'):
                Imageclassifier(uploaded_img_file)

            #Prediction 2 For ObjectDetection
            if st.button('🔍 Analyze for Obstructions'):
                Objectdetection(uploaded_img_file)
        else:
            st.warning('Please Upload the Image to analyze the solar panel Health')
    else:
        st.subheader("ℹ️ About This App")
        st.markdown("""
        **SolarScan 🌞🔍** is an intelligent object detection as well image classfier app designed to analyze solar panels and identify common issues.

        ### 🧠 Detectable Conditions
        - ✅ Clean
        - 🌫️ Dusty
        - 💩 Bird Drop
        - ⚡ Electrical Damage
        - 🔨 Physical Damage
        - ❄️ Snow Covered

        ### 📸 How to Use:
        1. Select **'Detect Panel Condition'** in the sidebar.
        2. Upload a solar panel image.
        3. Click **'🔬 Identify Panel Condition'** to analyze.
        ###

        ---
        """)



if __name__=='__main__':
    main()

    
