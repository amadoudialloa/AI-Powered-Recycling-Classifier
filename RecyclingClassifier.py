# Import necessary libraries
import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import os

# Set the working directory to the root of your repository
root_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(root_dir)

# Function to load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    # Specify the relative path to the model file (assuming it's in the same directory)
    model_path = "./waste_classifier_model_with_augmentation.h5"
    try:
        loaded_model = tf.keras.models.load_model(model_path)
        return loaded_model
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None  # Return None in case of an error

# Load the model
model = load_model()

def classify_waste(image):
    if model is None:
        return "Unknown", 0.0  # Return default values in case of an error

    # Preprocess the image and perform classification here
    img_height, img_width = 224, 224
    
    # Preprocess the image (resize and normalize pixel values)
    image = image.resize((img_height, img_width))
    image_array = np.array(image) / 255.0  # Normalize pixel values

    # Use the loaded model for predictions
    class_names = ["can", "glass", "plastic"]
    prediction = model.predict(np.expand_dims(image_array, axis=0))
    predicted_class = class_names[np.argmax(prediction)]
    confidence_score = np.max(prediction)

    return predicted_class, confidence_score



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


