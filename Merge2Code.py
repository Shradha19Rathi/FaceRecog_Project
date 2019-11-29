import cv2
import numpy as np

name=input("Enter a name : ")

cam = cv2.VideoCapture(0)
cam.set(6, 620) # width
cam.set(6, 630) # height

img_array = []
img_gray = []
labels = []

count = 0

face_detector = cv2.CascadeClassifier('HaarCascade\\haarcascade_frontalface_default.xml')

while(True):

    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        # cv2.imwrite(os.path.join('temp\\', str(count)+ '.jpg' ), gray[y:y+h,x:x+w])
        # cv2.imshow('image', img)
        img_array.append(img)
        cv2.waitKey(60)
         
    if count >= 3:
        break

def detect_face(j, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('HaarCascade\\haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    if (len(faces) == 0):
        return None, None

    (x, y, w, h) = faces[0]    
    # cv2.imshow(f'image {j}', gray[y:y+w, x:x+h])
    img_gray.append(gray[y:y+w, x:x+h])
    return gray[y:y+w, x:x+h], faces[0]

for i in range(len(img_array)):
    detect_face(i, img_array[i])
    labels.append(i+1)

# print(len(img_array))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
         
face_recognizer.train(img_gray, np.array(labels))
face_recognizer.write('test-data\\' + "%s" %name +'.json')

cv2.waitKey(0)
cv2.destroyAllWindows()
