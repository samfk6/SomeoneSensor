#!/usr/bin/python
import cv2
import os

TARGET_DIR = "out/"

if not (os.path.exists(TARGET_DIR)):
    os.mkdir(TARGET_DIR)

capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
tmp = 0
while True:
    ret, frame = capture.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray,1.1,10)

    for (x,y,w,h) in face:
    	roi = frame[y:y+h, x:x+w]
	cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
	cv2.imwrite( TARGET_DIR + str(tmp)+".bmp", roi )
	tmp = tmp + 1

    cv2.imshow('frame',frame)
    if cv2.waitKey(10) == 27:
        break

capture.release()
cv2.destroyAllWindows()
