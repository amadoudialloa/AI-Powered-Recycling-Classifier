# Import necessary libraries
import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import sys  # Import the sys module

# Check if running locally based on a command-line argument
running_locally = "--local" in sys.argv

# Function to load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model("waste_classifier_model_with_augmentation.h5")

# Load the model
if running_locally:
    os.system("python train_model.py")  # Run the training script if running locally
    model = load_model()  # Load the trained model if running locally
else:
    model = load_model()  # Load the trained model during deployment


# ... (Rest of your Streamlit app code, including UI and classification logic)


# Define dataset path (the same as in train_model.py)
dataset_path = "/Users/alphadiallo/Desktop/DSProjects/Mecro/RecycleClassifier/recycle_classifier/materials"
img_height, img_width = 224, 224

# ... (Rest of your Streamlit app code, including UI and classification logic)
# Streamlit app
st.title("Waste Classification App")
st.write("""AI-Powered Recycling Classifier

Developed by: Alpha Diallo

Summary:

The AI-Powered Recycling Classifier is a groundbreaking solution designed to identify and classify waste materials, such as plastic, glass, and cans, using the latest advancements in artificial intelligence. This project streamlines the recycling process by automatically categorizing waste items, promoting sustainable practices, and contributing to environmental conservation.

Description:

Recycling plays a crucial role in reducing waste and conserving valuable resources. The AI-Powered Recycling Classifier project leverages state-of-the-art machine learning techniques to recognize and categorize recyclable materials, simplifying the recycling journey for individuals and organizations alike.

How It Works:

Image Recognition: Users upload images of waste materials, such as bottles, cans, or plastic containers, to the application.

AI Classification: My advanced AI model processes the uploaded images and classifies the waste material into predefined categories (e.g., plastic, glass, cans).

Instant Results: The application provides instant feedback, indicating the type of waste material and its recyclability status.

Key Features:

Automated Waste Classification

Environmental Sustainability

User-Friendly Interface

Promotion of Recycling Practices

Key Benefits:

Effortless Recycling

Increased Recycling Rates

Environmental Impact

Sustainable Living """)

st.write("Upload an image of waste, and I will classify it.")

# File upload widget
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Classify the uploaded image
    predicted_class, confidence_score = classify_waste(image)

    # Display the prediction and confidence score
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence Score: {confidence_score:.2f}")

