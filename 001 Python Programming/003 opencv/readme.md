# Basic Python Programming

## Using Image Library (openCV)

Image manipulation is an important task in preparing for a machine vision project. Resizing, cropping, and flipping are the most common operations. Sometimes, we need to add some noise to the image using gaussian blur etc. In this course, we are going to manipulate the given image with openCV library to put sunglasses on a person.

To do this, several libraries are required:

1. os : to read and write the files
2. numpy : to store image data within the numpy arrays
3. opencv : to manipulate the images
4. dlib : to recognize the facial landmarks

## Process

1. Read the given image (person.jpg) and make the photo look straight.
   1. Use rotate method.
2. Resize the given image into 50%.
3. Use openCV and dlib libraries to extract the facial landmarks from the given image.
   1. Link to download the landmark data:  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
4. Crop the area in the image that contains the face.
   1. Make sure that the cropped area contains the all of the facial landmarks.
5. Read the sunglasses image (sunglasses.png) and resize it to fit the face image that you cropped.
   1. Note that the sunglasses image is in png format (4 channels).
6. Put the sunglasses on a face (cropped image).
   1. Make sure that the transparent background of the sunglasses image is preserved.
   2. Hint : replace the pixel information at the desired position with the sunglasses image.

## Important codes

### Read and rotate the image

```python
import cv2

image = cv2.imread("person.jpg", cv2.IMREAD_COLOR)

height, width, channel = image.shape
matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 1)
dst = cv2.warpAffine(image, matrix, (width, height))

cv2.imshow("image", image)
cv2.imshow("dst", dst)
cv2.waitKey(0)
# You can close the window by pressing any key.
cv2.destroyAllWindows()
```

### Resize the image with given ratio
```python
import cv2

# load the image and show it
image = cv2.imread('person.png')
 
# when the desired width is 600
ratio = 600.0 / image.shape[1]
dim = (600, int(image.shape[0] * ratio))
 
# perform the actual resizing of the image
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
cv2.imshow("Resized", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Extract the facial landmarks using dlib and camera.
```python
import dlib
import cv2

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)
 
while True:
    # load the input image and convert it to grayscale
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # detect faces in the grayscale image
    rects = detector(gray, 0)
    
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
    
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    
    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
```

### Crop the image
```python
# x, y, w, h refers to the ROI for which the image is to be cropped. 
img = cv2.imread('person.jpg') 
cropped_img = img[y: y + h, x: x + w]
```

## Author
Hwanmoo Yong