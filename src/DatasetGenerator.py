import cv2

# Loading cascade classifier for face detection
cascadePath = "./opencv/data/haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Initializing Camera
camera = cv2.VideoCapture(0)

while True:
    # ret (True/False responsible for detecting whether there is an image)
    # frame (Our RGB image matrix from video stream)
    ret, frame = camera.read()
    # Converts our frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detects objects of different sizes in the input image.
    # The detected objects are returned as a list of rectangles.
    faces = faceCascade.detectMultiScale(gray)

    biggestIndex = 0
    faceArea = 0

    for i in range(len(faces)):

        newFaceArea = faces[i][2] * faces[i][3]

        if newFaceArea > faceArea:
            faceArea = newFaceArea
            biggestIndex = i

    x, y, w, h = faces[i]

    cropImage = frame[y-25:y+h+25, x-25:x+w+25]
    cropImage = cv2.resize(cropImage, (200, 200))

    print(frame)
    break
