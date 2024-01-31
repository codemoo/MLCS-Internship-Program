import cv2

image = cv2.imread("person.jpg", cv2.IMREAD_COLOR)

height, width, channel = image.shape
matrix = cv2.getRotationMatrix2D((width/2, height/2), 270, 1)
dst = cv2.warpAffine(image, matrix, (width, height))

# cv2.imshow("image", image)
cv2.imshow("dst", dst)
cv2.waitKey(0)
# You can close the window by pressing any key.
cv2.destroyAllWindows()