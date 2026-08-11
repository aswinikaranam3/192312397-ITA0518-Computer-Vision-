
# Install required libraries
!pip install opencv-python matplotlib scikit-image

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from skimage.filters import prewitt

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
# SOBEL OPERATOR
# -----------------------------
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

sobel_mag = np.sqrt(sobelx**2 + sobely**2)
sobel_dir = np.arctan2(sobely, sobelx)

# -----------------------------
# PREWITT OPERATOR
# -----------------------------
# Define kernels manually
kernelx = np.array([[-1,0,1],
                    [-1,0,1],
                    [-1,0,1]])

kernely = np.array([[-1,-1,-1],
                    [0,0,0],
                    [1,1,1]])

prewittx = cv2.filter2D(gray, -1, kernelx)
prewitty = cv2.filter2D(gray, -1, kernely)

prewitt_mag = np.sqrt(prewittx**2 + prewitty**2)
prewitt_dir = np.arctan2(prewitty, prewittx)

# -----------------------------
# Sample Gradient Values (Center Pixel)
# -----------------------------
h, w = gray.shape
cx, cy = h//2, w//2

print("------ SAMPLE GRADIENT VALUES ------")
print("SOBEL -> Gx:", sobelx[cx,cy], "Gy:", sobely[cx,cy], "Magnitude:", sobel_mag[cx,cy])
print("PREWITT -> Gx:", prewittx[cx,cy], "Gy:", prewitty[cx,cy], "Magnitude:", prewitt_mag[cx,cy])

# -----------------------------
# Normalize for display
# -----------------------------
sobel_disp = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX)
prewitt_disp = cv2.normalize(prewitt_mag, None, 0, 255, cv2.NORM_MINMAX)

# -----------------------------
# Display Results
# -----------------------------
titles = [
    "Original Image", "Grayscale",
    "Sobel Edge Magnitude", "Prewitt Edge Magnitude",
    "Sobel Direction", "Prewitt Direction"
]

images = [
    img, gray,
    sobel_disp, prewitt_disp,
    sobel_dir, prewitt_dir
]

plt.figure(figsize=(12,8))
for i in range(len(images)):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
