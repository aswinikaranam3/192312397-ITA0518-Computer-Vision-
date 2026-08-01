
import cv2
image = cv2.imread("C:\\Users\\sree\\Desktop\\OpenCV\\image1.jpg")
width = image.shape[1]
height = image.shape[0]
new_image = cv2.imread("C:\\Users\\girishma\\Desktop\\OpenCV\\image1.jpg"))
current_position = cv2.getWindowProperty("Original Image", cv2.WND_PROP_POSITION)
cv2.moveWindow("Original Image", current_position[0] + 100, current_position[1] + 100)
cv2.imshow("Original Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
