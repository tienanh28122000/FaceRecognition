import cv2

cascadePath = "./opencv/data/haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

camera = cv2.VideoCapture(0)
# camera = cv2.VideoCapture("./data/videos/markiyan1.mp4")


while True:
    ret, frame = camera.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(10, 10),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

    # for (x, y, w, h) in faces:
    #     cropImage = frame[y-25:y+h+25, x-25:x+w+25]
    #     cropImage = cv2.resize(cropImage, (200, 200))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y - 10), (x + w, y + h + 10), (255, 255, 255), 2)
            cv2.putText(frame, "Vitaliy Vorobyov", (x + w, y + h + 10), 0,0.3,(0,255,0))

        cv2.imshow("Live Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Camera isn't connected")
        break

# while True:
#     ret, frame = camera.read()
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     faces = faceCascade.detectMultiScale(
#         gray,
#         scaleFactor=1.3,
#         minNeighbors=5,
#         minSize=(10, 10),
#         flags=cv2.CASCADE_SCALE_IMAGE
#     )
