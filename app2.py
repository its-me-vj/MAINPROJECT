import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import sys

# Add model directory to Python path
sys.path.append("C:/Users/hp/Desktop/Crowd Density/Crowd Density/models")
from csrnet import CSRNet  # Import CSRNet from the correct path

# Load the model
model_path = Path("C:/Users/hp/Desktop/Crowd Density/Crowd Density/crowd_counting_model/weights.pth")
model = CSRNet()

# Load state dict and filter mismatched layers
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model_dict = model.state_dict()
filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}

# Update the model's state_dict with the filtered weights
model_dict.update(filtered_state_dict)
model.load_state_dict(model_dict)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Streamlit UI setup
st.title("Crowd Density Estimation")
st.write("Upload an image to estimate the crowd count.")

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    image = transform(image).unsqueeze(0)

    # Predict the crowd count
    with torch.no_grad():
        output = model(image)
        count = int(output.sum().item())  # Sum the pixel values to estimate crowd count

    # Display result
    st.write(f"Estimated Crowd Count: {count}")
