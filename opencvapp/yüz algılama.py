import cv2  
import numpy as np  

vid = cv2.VideoCapture(0)

yuz_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

if not vid.isOpened():
    print("Kamera açılamadı")
    exit()

while True:
    ret, frame = vid.read()

    frame = cv2.flip(frame, 1) 
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    yuzler = yuz_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in yuzler:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (85, 255, 0), 3)

    if not ret:
        print("Görüntü alınamadı")
        break

    cv2.imshow('Kamera Görüntüsü', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()

cv2.destroyAllWindows()
