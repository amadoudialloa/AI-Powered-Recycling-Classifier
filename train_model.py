# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define dataset path, image dimensions, batch size, etc. (use a relative path)
dataset_path = os.path.join(current_dir, "materials")
img_height, img_width = 224, 224
batch_size = 20

# Create an ImageDataGenerator with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
)

# Load and preprocess the dataset
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
)

validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
)

# Create a base model (MobileNetV2) and add custom classification layers
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(3, activation="softmax")(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(
    train_generator,
    epochs=1,  # Adjust as needed
    validation_data=validation_generator,
    steps_per_epoch=None,
    validation_steps=None,
)

# Save the trained model
model.save("waste_classifier_model_with_augmentation.h5")


