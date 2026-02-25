"""
=============================================================================
Offroad Image Classification using InceptionV3 Transfer Learning
=============================================================================
Author      : [Your Team Name]
Date        : 2026
Framework   : TensorFlow 2.20 / Keras
GPU         : NVIDIA Tesla T4 (Google Colab)
Task        : Binary image classification (Offroad vs Non-Offroad)
=============================================================================

Description:
    This script builds, trains, and evaluates a binary image classifier
    using InceptionV3 as a frozen feature extractor (pretrained on ImageNet).
    A custom Dense classification head is added on top and trained on the
    Offroad Segmentation dataset.

Results:
    - Final Val Accuracy : 99.20%
    - Final Val IoU      : 0.9817
    - Final Val Dice     : 0.9908
    - Final Val Loss     : 0.1080
=============================================================================
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from glob import glob

import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model


# =============================================================================
# 2. CONFIGURATION
# =============================================================================

# Input image dimensions (InceptionV3 default: 299x299, but 224x224 also works)
IMAGE_SIZE = [224, 224]

# Dataset paths (update these paths to match your environment)
TRAIN_PATH = '/content/drive/MyDrive/hackathon/Offroad_Segmentation_Training_Dataset/train'
VALID_PATH = '/content/drive/MyDrive/hackathon/Offroad_Segmentation_Training_Dataset/val'
TEST_PATH  = '/content/drive/MyDrive/hackathon/Offroad_Segmentation_testImages'

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS     = 10


# =============================================================================
# 3. GPU CHECK
# =============================================================================

print("Available GPUs:", tf.config.list_physical_devices('GPU'))
# Expected output: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]


# =============================================================================
# 4. LOAD PRETRAINED BASE MODEL (InceptionV3)
# =============================================================================

# Load InceptionV3 without the top classification layers
# weights='imagenet' loads pretrained ImageNet weights
# include_top=False removes the final Dense layers so we can add our own
inception = InceptionV3(
    input_shape=IMAGE_SIZE + [3],   # (224, 224, 3)
    weights='imagenet',
    include_top=False
)

# Freeze all layers in the base model — we only train the new classification head
for layer in inception.layers:
    layer.trainable = False

print(f"Base model loaded. Total layers: {len(inception.layers)}")


# =============================================================================
# 5. BUILD CUSTOM CLASSIFICATION HEAD
# =============================================================================

# Determine number of output classes from training folder structure
folders = glob(TRAIN_PATH + '/*')
num_classes = len(folders)
print(f"Number of classes: {num_classes}")

# Add custom head on top of InceptionV3 output
x = Flatten()(inception.output)                              # Flatten feature maps → (None, 51200)
prediction = Dense(num_classes, activation='softmax')(x)    # Final classification layer

# Assemble the full model
model = Model(inputs=inception.input, outputs=prediction)

# Display model summary
model.summary()
# Trainable params    : ~102,402  (classification head)
# Non-trainable params: ~21,802,784 (frozen InceptionV3)


# =============================================================================
# 6. COMPILE MODEL
# =============================================================================

model.compile(
    loss='categorical_crossentropy',    # Multi-class cross-entropy loss
    optimizer='adam',                    # Adaptive Moment Estimation optimizer
    metrics=[
        'accuracy',
        tf.keras.metrics.MeanIoU(num_classes=num_classes)  # IoU for segmentation quality
    ]
)

print("Model compiled successfully.")


# =============================================================================
# 7. DATA PREPROCESSING & AUGMENTATION
# =============================================================================

# Training data generator with augmentation to reduce overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,         # Normalize pixel values to [0, 1]
    shear_range=0.2,        # Random shear transformation
    zoom_range=0.2,         # Random zoom
    horizontal_flip=True    # Random horizontal flip
)

# Test/Validation data generator — only normalization, no augmentation
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training images from directory
training_set = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(224, 224),     # Resize all images to 224x224
    batch_size=BATCH_SIZE,
    class_mode='categorical'    # One-hot encoded labels for multi-class
)
# Expected: Found 5716 images belonging to 2 classes.

# Load test/validation images from directory
test_set = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
# Expected: Found 2003 images belonging to 2 classes.

print(f"Class indices: {training_set.class_indices}")


# =============================================================================
# 8. TRAIN THE MODEL
# =============================================================================

print("\nStarting training...")

r = model.fit(
    training_set,
    validation_data=test_set,
    epochs=EPOCHS,
    steps_per_epoch=len(training_set),      # ~179 steps (5716 / 32)
    validation_steps=len(test_set)          # ~63 steps (2003 / 32)
)

print("\nTraining complete.")


# =============================================================================
# 9. PLOT TRAINING CURVES
# =============================================================================

# --- Loss Curves ---
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(r.history['loss'], label='Train Loss')
plt.plot(r.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# --- Accuracy Curves ---
plt.subplot(1, 2, 2)
plt.plot(r.history['accuracy'], label='Train Accuracy')
plt.plot(r.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()
print("Training curves saved to 'training_curves.png'")


# =============================================================================
# 10. SAVE THE MODEL
# =============================================================================

model.save('model_inception.h5')
print("Model saved as 'model_inception.h5'")

# NOTE: Keras recommends the native .keras format going forward:
# model.save('model_inception.keras')


# =============================================================================
# 11. GENERATE PREDICTIONS ON TEST SET
# =============================================================================

# Get raw softmax probability outputs for all test images
y_pred_probs = model.predict(test_set)
print(f"Raw predictions shape: {y_pred_probs.shape}")
# Example: [[7.76e-23, 1.0], [1.0, 0.0], ...]  — one probability per class

# Convert softmax probabilities to class indices (0 or 1)
y_pred = np.argmax(y_pred_probs, axis=1)
print(f"Predicted class indices (first 10): {y_pred[:10]}")


# =============================================================================
# 12. CALCULATE IoU SCORE
# =============================================================================

# True labels from test generator
y_true = test_set.labels

# Use TensorFlow's MeanIoU metric
m = tf.keras.metrics.MeanIoU(num_classes=num_classes)
m.update_state(y_true, y_pred)   # Both are integer class labels

iou_score = m.result().numpy()
print(f"\nIntersection over Union (IoU) Score: {iou_score:.4f}")


# =============================================================================
# 13. FINAL EVALUATION
# =============================================================================

# Reload model (demonstrates save/load works correctly)
model = load_model('model_inception.h5')

# Recompile after loading to restore metrics
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=[
        'accuracy',
        tf.keras.metrics.MeanIoU(num_classes=2)
    ]
)

# Run full evaluation on test set
results = model.evaluate(test_set, verbose=1)

final_loss     = results[0]
final_accuracy = results[1]
final_iou      = results[2]

# Derive Dice coefficient from IoU: Dice = (2 * IoU) / (1 + IoU)
final_dice = (2 * final_iou) / (1 + final_iou)

# Print summary
print("\n" + "="*50)
print("       FINAL EVALUATION RESULTS")
print("="*50)
print(f"  Val Loss     : {final_loss:.4f}")
print(f"  Val Accuracy : {final_accuracy:.4f}  ({final_accuracy*100:.2f}%)")
print(f"  Val IoU      : {final_iou:.4f}")
print(f"  Val Dice     : {final_dice:.4f}")
print("="*50)


# =============================================================================
# 14. INFERENCE ON A SINGLE IMAGE (optional utility)
# =============================================================================

def predict_single_image(image_path, model, image_size=(224, 224)):
    """
    Run inference on a single image file.

    Args:
        image_path (str): Path to the image file.
        model      : Loaded Keras model.
        image_size (tuple): Target resize dimensions (width, height).

    Returns:
        predicted_class (int): Predicted class index.
        confidence (float): Softmax confidence for the predicted class.
    """
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    # Load and preprocess image
    img = load_img(image_path, target_size=image_size)
    img_array = img_to_array(img) / 255.0           # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension → (1, 224, 224, 3)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class]

    return predicted_class, confidence


# Example usage:
# class_idx, conf = predict_single_image('/path/to/image.jpg', model)
# print(f"Predicted Class: {class_idx}, Confidence: {conf:.4f}")
