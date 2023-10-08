# Import necessary libraries
import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import sys

# Check if running locally based on a command-line argument
running_locally = "--local" in sys.argv

# Function to load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = "waste_classifier_model_with_augmentation.h5"
    try:
        loaded_model = tf.keras.models.load_model(model_path)
        return loaded_model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None  # Return None if there is an error

# Function to preprocess and classify the uploaded image
def classify_waste(image):
    try:
        # Load the model
        model = load_model()

        if model is not None:
            # Preprocess the image
            image = image.resize((224, 224))  # Resize to match the model's input size
            image_array = np.array(image) / 255.0  # Normalize pixel values

            # Make predictions using the model
            class_names = ["can", "glass", "plastic"]
            prediction = model.predict(np.expand_dims(image_array, axis=0))

            # Get predicted class and confidence score
            predicted_class = class_names[np.argmax(prediction)]
            confidence_score = np.max(prediction)

            return predicted_class, confidence_score

    except Exception as e:
        st.error(f"Error classifying the image: {e}")

    return "Unknown", 0.0  # Return "Unknown" and confidence score 0.0 in case of an error


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

