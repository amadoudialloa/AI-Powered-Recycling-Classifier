import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Define the dataset path
dataset_path = "/Users/alphadiallo/Desktop/DSProjects/Mecro/RecycleClassifier/recycle_classifier/materials"

# Define image dimensions
img_height, img_width = 224, 224
batch_size = 40

# Create an ImageDataGenerator with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize pixel values
    rotation_range=20,  # Randomly rotate images by up to 20 degrees
    width_shift_range=0.2,  # Randomly shift image width by up to 20%
    height_shift_range=0.2,  # Randomly shift image height by up to 20%
    shear_range=0.2,  # Shear transformations
    zoom_range=0.2,  # Randomly zoom in by up to 20%
    horizontal_flip=True,  # Randomly flip images horizontally
    validation_split=0.2,  # Split data into training and validation
)

# Load and preprocess the dataset
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",  # Training split
)

validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",  # Validation split
)

# Create a base model (MobileNetV2) and add custom classification layers
base_model = MobileNetV2(weights="imagenet", include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(3, activation="softmax")(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# Train the model using all available data
model.fit(
    train_generator,
    epochs=30,  # Adjust as needed
    validation_data=validation_generator,
    steps_per_epoch=None,
    validation_steps=None,
)

# Save the trained model
model.save("waste_classifier_model_with_augmentation.h5")

# Function to preprocess and classify the uploaded image
def classify_waste(image):
    # Load the trained model
    model = tf.keras.models.load_model("waste_classifier_model_with_augmentation.h5")

    # Preprocess the image (resize and normalize pixel values)
    image = image.resize((img_height, img_width))
    image = np.array(image) / 255.0  # Normalize pixel values

    # Make predictions using the model
    class_names = ["can", "glass", "plastic"]
    prediction = model.predict(np.expand_dims(image, axis=0))
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
