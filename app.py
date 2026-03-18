import streamlit as st
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.utils import load_data
from backend.client import Client
from backend.federated import federated_training
from PIL import Image
import torch

st.title("Federated Learning Image Classifier")

uploaded_file = st.file_uploader("Upload Image", type=["png","jpg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('L')
    st.image(image, width=200)

    loaders = load_data()
    clients = [Client(loader) for loader in loaders]

    model = federated_training(clients)

    img_tensor = torch.tensor(image.resize((28,28))).float().unsqueeze(0).unsqueeze(0)

    output = model(img_tensor)
    prediction = torch.argmax(output, dim=1).item()

    st.success(f"Prediction: {prediction}")
