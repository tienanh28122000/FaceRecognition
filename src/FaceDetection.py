import cv2

imagePath = "./data/faces-data/1/100001.jpg"
cascPath = "./opencv/data/haarcascades/haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascPath)

image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(gray)

print(faces)

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y-10), (x+w, y+h+10), (255, 255, 255), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
