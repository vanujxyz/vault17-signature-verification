import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import imghdr  # Added to check if the file is an image

# Paths to your dataset
real_signatures_path = 'dataset/original_signs/'
forged_signatures_path = 'dataset/forged_signs/'

# Image size for resizing
img_size = (64, 64)

# Function to load and preprocess images
def load_images(signatures_path, label):
    images = []
    labels = []
    for img_name in os.listdir(signatures_path):
        img_path = os.path.join(signatures_path, img_name)
        
        # Check if the file is an image
        if imghdr.what(img_path) is None:
            continue  # Skip non-image files
        
        # Read image in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Warning: Unable to read image {img_name}")
            continue  # Skip unreadable images

        # Resize the image
        img_resized = cv2.resize(img, img_size)

        # Normalize pixel values to be between 0 and 1
        img_normalized = img_resized.astype('float32') / 255.0

        # Append to images list and assign the label (1 for real, 0 for forged)
        images.append(img_normalized)
        labels.append(label)

    return images, labels

# Load real and forged signature images
real_images, real_labels = load_images(real_signatures_path, label=1)
forged_images, forged_labels = load_images(forged_signatures_path, label=0)

# Combine real and forged images and labels
images = real_images + forged_images
labels = real_labels + forged_labels

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Flatten the images to 1D vectors (for classical ML models)
images_flat = images.reshape(images.shape[0], -1)

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(images_flat, labels, test_size=0.2, random_state=42)

# Print the shape of the resulting datasets
print(f"Training Set: {X_train.shape}, Test Set: {X_test.shape}")
