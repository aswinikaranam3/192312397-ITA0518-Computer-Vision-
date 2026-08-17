import cv2
import numpy as np
image = cv2.imread("C:\\Users\\girishma\\Desktop\\OpenCV\\image1.jpg")
kernel = np.array([[0, 1, 0],
                   [1, -8, 1],
                   [0, 1, 0]])
sharpened = cv2.filter2D(image, -1, kernel)
cv2.imshow('Original', image)
cv2.imshow('Sharpened.jpg', sharpened)
