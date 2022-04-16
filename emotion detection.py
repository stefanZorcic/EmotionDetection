import PIL.ImageGrab
import cv2
import time
import numpy as np

from tensorflow import keras
model = keras.models.load_model('finalModel')

def takeImage(x1,y1,x2,y2):
    x1=int(x1)
    x2=int(x2)
    y1=int(y1)
    y2=int(y2)
    im1 = PIL.ImageGrab.grab()
    im = im1.crop((x1,y1,x2,y2))
    im.save("temp.png")

def runAnalysis():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv2.imread('temp.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        crop_img1 = img[y:y + h, x:x + w]
        crop_img = cv2.cvtColor(crop_img1, cv2.COLOR_BGR2GRAY)
        crop_img = crop_img/255
        image = cv2.resize(crop_img, (50, 50))
        face = np.array(image, dtype="float32").reshape(-1, 50, 50, 1)
        vals = (model.predict(face))
        emotions = ["anger", "contempt", "disgust", "fear", "happiness", "neutrality", "sadness", "surprise"]
        temparr=[]
        for i in range(len(vals[0])):
            temparr.append([(vals[0][i]*100),emotions[i]])
        temparr.sort(reverse=True)
        for i in range(8):
            print(temparr[i][1],temparr[i][0])
        print()
        print()
        cv2.imshow('crop_img1', crop_img)
        cv2.waitKey()


while True:
    print(">>> ",end="")
    command1  = str(input())
    if (command1=="exit"):
        break;
    command = list(map(str,command1.split()))
    if (command[0]=="analysis"):
        takeImage(command[1], command[2], command[3], command[4])
        runAnalysis()
        print()
    else:
        print("Please use one of the following commands: ")
        print("exit - to exit the program")
        print("analysis x1 y1 x2 y2 - to run analysis on a section of your screen")