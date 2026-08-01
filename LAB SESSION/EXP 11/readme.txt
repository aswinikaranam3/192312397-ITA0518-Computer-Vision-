import cv2
import numpy as np
img = cv2.imread("C:\\Users\\girishma\\Desktop\\OpenCV\\image1.jpg")
rows, cols = img.shape[:2]
M = np.float32([[1, 0, 1000], [0, 1, 500]])
affine_img = cv2.warpAffine(img, M, (cols, rows))
cv2.imwrite('Affine_Transformed.jpg', affine_img)
