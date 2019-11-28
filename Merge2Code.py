import cv2
import os
import numpy as np
import shutil

Name=str(input("Enter a name : "))

cam = cv2.VideoCapture(1)
cam.set(6, 620) # width
cam.set(6, 630) # height

path = 'C:\\Users\\Shradha\\Desktop\\New_Database\\'
srcdirs = os.path.join(path,  "%s" %Name)

oldpath = 'C:\\Users\\Shradha\\Desktop\\Old_Database\\'
dstdirs = os.path.join(oldpath,  "%s" %Name)

feature_file = 'C:\\Users\\Shradha\\Desktop\\YML_File\\'

try:

    os.mkdir(path)
    os.mkdir(srcdirs)
    os.mkdir(dstdirs)
    os.mkdir(feature_file)


    
except:
    print("folder exsits")
if not os.path.exists(srcdirs):
    os.makedirs(srcdirs)    
    
count = 0
face_detector = cv2.CascadeClassifier('C:\\Users\\Shradha\\Desktop\\opencv_haarcascade_frontalface_default\\haarcascade_frontalface_default.xml')


while(True):

    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        cv2.imwrite(os.path.join(srcdirs, str(count)+ '.jpg' ), gray[y:y+h,x:x+w])
        cv2.imshow('image', img)
        cv2.waitKey(60)
    k = cv2.waitKey(40) & 0xff 
    if k == 27:
        break
    elif count >= 5:
        break

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('E:\\IOT DATA\\opencv-face-recognition-python-master1\\opencv-face-recognition-python-master\\HaarCascade\\haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    if (len(faces) == 0):
        return None, None
    
    (x, y, w, h) = faces[0]
    
    return gray[y:y+w, x:x+h], faces[0]

def prepare_training_data(data_folder_path):
    
    dirs = os.listdir(data_folder_path)
    
    faces = []
    labels = []
    
    for dir_name in dirs:
        label=1
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        
        
        for image_name in subject_images_names:
            
            if image_name.startswith("."):
                continue
            
            image_path = subject_dir_path + "/" + image_name
            print("aaaddd:",image_path)
            image = cv2.imread(image_path)
            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(100)
            
            face, rect = detect_face(image)
            print(" Path_Images: ",image_name)
            print("AAA   ", face)

            
            if face is not None:
                faces.append(face)
                labels.append(label)
                
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels

print("Preparing data...")
faces, labels = prepare_training_data("New_Database")
print("Data prepared")
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))
face_recognizer.write( feature_file + "%s" %Name +'.yml')
shutil.move(srcdirs,dstdirs)

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()





