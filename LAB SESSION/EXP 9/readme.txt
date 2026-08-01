import cv2
img = cv2.imread("C:\\Users\\girishma\\Desktop\\OpenCV\\image1.jpg"))
# Rotate clockwise
rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
# Rotate counterclockwise
rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imwrite("rotated_image.jpg",rotated_img)
