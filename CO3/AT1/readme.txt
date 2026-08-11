# Install required libraries (if needed)
!pip install opencv-python matplotlib scikit-image

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from skimage.filters import prewitt

# Upload image
uploaded = files.upload()

# Read image
image_path = list(uploaded.keys())[0]
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Add Gaussian Noise
noise = np.random.normal(0, 25, gray.shape).astype(np.uint8)
noisy_img = cv2.add(gray, noise)

# -----------------------------
# Smoothing Techniques
# -----------------------------
mean = cv2.blur(noisy_img, (5,5))
gaussian = cv2.GaussianBlur(noisy_img, (5,5), 0)
median = cv2.medianBlur(noisy_img, 5)
bilateral = cv2.bilateralFilter(noisy_img, 9, 75, 75)

# -----------------------------
# Edge Detection
# -----------------------------
# Canny
canny_noisy = cv2.Canny(noisy_img, 100, 200)
canny_gaussian = cv2.Canny(gaussian, 100, 200)

# Sobel
sobelx = cv2.Sobel(gaussian, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gaussian, cv2.CV_64F, 0, 1, ksize=5)
sobel = cv2.magnitude(sobelx, sobely)

# Prewitt
prewitt_edges = prewitt(gaussian)

# -----------------------------
# Display Results
# -----------------------------
titles = [
    "Original Image", "Grayscale", "Noisy Image",
    "Mean Filter", "Gaussian Filter", "Median Filter", "Bilateral Filter",
    "Canny (No Noise Removal)", "Canny (After Gaussian)",
    "Sobel Edge", "Prewitt Edge"
]

images = [
    img, gray, noisy_img,
    mean, gaussian, median, bilateral,
    canny_noisy, canny_gaussian,
    sobel, prewitt_edges
]

plt.figure(figsize=(15,10))
for i in range(len(images)):
    plt.subplot(3,4,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
