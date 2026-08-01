import cv2
import numpy as np
img = cv2.imread("C:\\Users\\girishma\\Desktop\\OpenCV\\image1.jpg"))
rows, cols = img.shape[:2]
src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
dst_points = np.float32([[0, 0], [cols - 1, 0], [int(0.33*cols), rows - 1], [int(0.66*cols), rows - 1]])
M = cv2.getPerspectiveTransform(src_points, dst_points)
perspective_img = cv2.warpPerspective(img, M, (cols, rows))
cv2.imwrite('Perspective_Transformed_Image.jpg', perspective_img)
