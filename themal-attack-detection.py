import numpy as np
import cv2

frame = cv2.imread("./data/COLOR/USER/USUARIO_057.jpg")
frame = cv2.imread("./data/COLOR/attack_01/spoof_057.jpg")

dim = tuple((np.array([frame.shape[1], frame.shape[0]]) / 4).astype(np.int))

frame = cv2.resize(frame, dim)

faceCascade = cv2.CascadeClassifier("./classifiers/haarcascade_frontalface_default.xml")
rects_face = faceCascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(350, 350),
                                     flags=cv2.CASCADE_SCALE_IMAGE)

for r in rects_face:
    cv2.rectangle(frame, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (0, 255, 0), 3)

imgGrey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cv2.imshow("daasdsdasdasa", imgGrey)
_, thrash = cv2.threshold(imgGrey, 215, 255, cv2.THRESH_BINARY)
cv2.imshow("dasdasdasa", thrash)
contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)

#user = cv2.cvtColor(user, cv2.COLOR_BGR2GRAY)
cv2.imshow("dasdasd", frame)

cv2.waitKey()

