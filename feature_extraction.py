import cv2
import numpy as np
import os
import imghdr
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Paths to dataset
real_signatures_path = 'dataset/original_signs/'
forged_signatures_path = 'dataset/forged_signs/'

# Image size
img_size = (64, 64)

# Function to extract features from an image
def extract_features(img_path):
    # Check if the file is an image
    if imghdr.what(img_path) is None:
        return None
    
    # Read image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None  # Skip unreadable images
    
    # Resize the image
    img = cv2.resize(img, img_size)
    
    # Normalize pixel values
    img = img.astype('float32') / 255.0

    # Feature 1: Mean & Standard Deviation of Pixel Intensity
    mean_intensity = np.mean(img)
    std_intensity = np.std(img)

    # Feature 2: Edge Detection using Sobel Filter
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
    edge_mean = np.mean(edge_magnitude)

    # Feature 3: Contour Features (Area & Perimeter)
    _, thresh = cv2.threshold(img, 0.5, 1.0, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours((thresh * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        contour_perimeter = cv2.arcLength(largest_contour, True)
    else:
        contour_area = 0
        contour_perimeter = 0

    return [mean_intensity, std_intensity, edge_mean, contour_area, contour_perimeter]

# Function to process dataset and create feature dataframe
def process_dataset(path, label):
    data = []
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        features = extract_features(img_path)
        if features:
            data.append(features + [label])  # Append label to feature list
    return data

# Process real and forged signatures
real_data = process_dataset(real_signatures_path, label=1)
forged_data = process_dataset(forged_signatures_path, label=0)

# Create DataFrame
columns = ["Mean Intensity", "Std Intensity", "Edge Mean", "Contour Area", "Contour Perimeter", "Label"]
df = pd.DataFrame(real_data + forged_data, columns=columns)

# Save as CSV
df.to_csv("signature_features.csv", index=False)

# Split data into training and test sets
X = df.drop(columns=["Label"])
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Feature extraction complete. Dataset saved as 'signature_features.csv'.")
print(f"Training Set: {X_train.shape}, Test Set: {X_test.shape}")
