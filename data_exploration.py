import cv2
import matplotlib.pyplot as plt
import os

# Paths to your dataset
real_signatures_path = 'dataset/original_signs/'
forged_signatures_path = 'dataset/forged_signs/'

# Function to display images
def display_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap='gray')
    plt.title(f"Signature: {image_path}")
    plt.axis('off')
    plt.show()

# Display one real signature
real_image = os.path.join(real_signatures_path, os.listdir(real_signatures_path)[0])  # First real signature
display_image(real_image)

# Display one forged signature
forged_image = os.path.join(forged_signatures_path, os.listdir(forged_signatures_path)[0])  # First forged signature
display_image(forged_image)

print(f"Real Signatures Count: {len(os.listdir(real_signatures_path))}")
print(f"Forged Signatures Count: {len(os.listdir(forged_signatures_path))}")
