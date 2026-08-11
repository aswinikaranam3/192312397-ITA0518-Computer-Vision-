# Install required libraries
!pip install opencv-python matplotlib

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# -----------------------------
# Upload Image
# -----------------------------
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

# Read image
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# -----------------------------
# Define scales
# -----------------------------
scales = [0.5, 0.75, 1.0, 1.5, 2.0]

# ORB detector
orb = cv2.ORB_create(nfeatures=1000)

results = []
feature_counts = []

# -----------------------------
# Process each scale
# -----------------------------
for scale in scales:
    resized = cv2.resize(gray, None, fx=scale, fy=scale)
    
    keypoints, descriptors = orb.detectAndCompute(resized, None)
    
    img_kp = cv2.drawKeypoints(resized, keypoints, None, color=(0,255,0))
    
    results.append(img_kp)
    feature_counts.append(len(keypoints))

# -----------------------------
# Display Results
# -----------------------------
plt.figure(figsize=(15,6))

for i, scale in enumerate(scales):
    plt.subplot(1,5,i+1)
    plt.imshow(results[i], cmap='gray')
    plt.title(f"Scale {scale}\nFeatures: {feature_counts[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()

# -----------------------------
# Plot Feature Count vs Scale
# -----------------------------
plt.figure()
plt.plot(scales, feature_counts, marker='o')
plt.xlabel("Scale Factor")
plt.ylabel("Number of Features")
plt.title("Effect of Scaling on Feature Detection")
plt.grid()
plt.show()

# -----------------------------
# Print Results Table
# -----------------------------
print("------ FEATURE COUNT TABLE ------")
for i in range(len(scales)):
    print(f"Scale {scales[i]} -> Features Detected: {feature_counts[i]}")
