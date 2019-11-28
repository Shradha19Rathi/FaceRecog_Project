import cv2
import os
import numpy as np

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('FarImages_Poonam_G.yml')
#subjects = ["", "Shradha",  "Mrunal", "Sanket"]

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('E:\\IOT DATA\\opencv-face-recognition-python-master1\\opencv-face-recognition-python-master\\HaarCascade\\haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    if (len(faces) == 0):
        return None, None
    
    (x, y, w, h) = faces[0]
    
    return gray[y:y+w, x:x+h], faces[0]


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (255,0,0), 2)

def predict(test_img):
    img = test_img.copy()
  
    face, rect = detect_face(img)

    cv2.waitKey(20)
  
    label, confidence= face_recognizer.predict(face)
    
    print("Image_Confidence :", confidence)
    
    label_text = subjects[label]
    print("Heyy!! ", label_text)
    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1]-5)
    
    return img
onoff=True
while(onoff):
    webcam = cv2.VideoCapture(1)
    check, frame = webcam.read()
    print(check)
    resized_img = cv2.resize(frame, (800, 500))
    cv2.imshow("Capturing", resized_img)

    cv2.waitKey(15)
    try:
        predicted_img1 = predict(resized_img)
        print("Inside")
        cv2.imshow(subjects[1], cv2.resize(predicted_img1, (400, 500)))
        cv2.waitKey(20)
        onoff=False
    except:
        print("Image not proper Plzz try again!! ")
        cv2.waitKey(40)
