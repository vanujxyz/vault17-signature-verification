import cv2
import numpy as np
import joblib

# Load the trained model
model = joblib.load("signature_verification_model.pkl")

# Function to extract features from a single image
def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))

    # Compute features
    mean_intensity = np.mean(img)
    std_intensity = np.std(img)
    edges = cv2.Canny(img, 100, 200)
    edge_mean = np.mean(edges)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_area = sum(cv2.contourArea(c) for c in contours)
    contour_perimeter = sum(cv2.arcLength(c, True) for c in contours)  # ADD THIS FEATURE

    return np.array([[mean_intensity, std_intensity, edge_mean, contour_area, contour_perimeter]])

# Test the model on a new signature
test_image_path = "jeff-bezoss-signature-signaturely-image.png"  # Change this!
features = extract_features(test_image_path)
prediction = model.predict(features)[0]

# Print result
if prediction == 1:
    print(f"{test_image_path} → Real Signature ✅")
else:
    print(f"{test_image_path} → Forged Signature ❌")
