from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from tkinter import *
import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

window = Tk()
window.title("Mask Detector")
window.geometry('600x200')

tab_control = ttk.Notebook(window)
tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab3=ttk.Frame(tab_control)
tab_control.add(tab1, text='Görüntüden Maske Tespiti')
tab_control.add(tab2, text='Video Maske Tespiti')
tab_control.add(tab3,text='Kamera Maske Tespiti')

lbl1 = Label(tab1, text= 'Dosya Konumu :    ',fg='blue')
txt = Entry(tab1,width=50)

txt.grid(column=1, row=0)
lbl1.grid(column=0, row=0)


def detectImageMask():#görüntü belleğe alır
    image = cv2.imread(txt.get())
    # image=cv2.resize(image,(400,400))
    cascPath = os.path.dirname(
        cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    model = load_model("mask_recog_ver2.h5")

    gray2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    yuzler = faceCascade.detectMultiScale(gray2,
                                          scaleFactor=1.1,
                                          minNeighbors=5,
                                          minSize=(60, 60),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    faces_list = []
    preds = []
    for (x, y, w, h) in yuzler:
        face_frame = image[y:y + h, x:x + w]
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        face_frame = cv2.resize(face_frame, (224, 224))
        face_frame = img_to_array(face_frame)
        face_frame = np.expand_dims(face_frame, axis=0)
        face_frame = preprocess_input(face_frame)
        faces_list.append(face_frame)
        if len(faces_list) > 0:
            preds = model.predict(faces_list)
        for pred in preds:
            (mask, withoutMask) = pred
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        cv2.putText(image, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        # Display the resulting frame
    cv2.imshow('Görüntü', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def showFile():#görüntüyü gösterir
    image=cv2.imread(txt.get())
    cv2.imshow('Görüntü',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def showVideo():
    cap = cv2.VideoCapture(txt2.get())

    while (cap.isOpened()):
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('Video', gray)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def detectCamMask():#görüntüyü kaydeder
    cascPath = os.path.dirname(
        cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    model = load_model("mask_recog_ver2.h5")

    video_capture = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,
                                             scaleFactor=1.1,
                                             minNeighbors=5,
                                             minSize=(60, 60),
                                             flags=cv2.CASCADE_SCALE_IMAGE)
        faces_list = []
        preds = []
        for (x, y, w, h) in faces:
            face_frame = frame[y:y + h, x:x + w]
            face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
            face_frame = cv2.resize(face_frame, (224, 224))
            face_frame = img_to_array(face_frame)
            face_frame = np.expand_dims(face_frame, axis=0)
            face_frame = preprocess_input(face_frame)
            faces_list.append(face_frame)
            if len(faces_list) > 0:
                preds = model.predict(faces_list)
            for pred in preds:
                (mask, withoutMask) = pred
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            # Display the resulting frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(1):
            break
    video_capture.release()
    cv2.destroyAllWindows()

def detectVideoMask():
    cascPath = os.path.dirname(
        cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    model = load_model("mask_recog_ver2.h5")

    video_capture = cv2.VideoCapture(txt2.get())
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,
                                             scaleFactor=1.1,
                                             minNeighbors=5,
                                             minSize=(60, 60),
                                             flags=cv2.CASCADE_SCALE_IMAGE)
        faces_list = []
        preds = []
        for (x, y, w, h) in faces:
            face_frame = frame[y:y + h, x:x + w]
            face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
            face_frame = cv2.resize(face_frame, (224, 224))
            face_frame = img_to_array(face_frame)
            face_frame = np.expand_dims(face_frame, axis=0)
            face_frame = preprocess_input(face_frame)
            faces_list.append(face_frame)
            if len(faces_list) > 0:
                preds = model.predict(faces_list)
            for pred in preds:
                (mask, withoutMask) = pred
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            # Display the resulting frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()



btn = Button(tab1,height=5,width=15, text="Maske Tespit Et",compound='c', command=detectImageMask)
btn.grid(column=1, row=2)
btn2 = Button(tab1, text="Görüntü Göster", command=showFile)
btn2.grid(column=2, row=0)

lbl2 = Label(tab2, text= 'Dosya Konumu :    ',fg='blue')
txt2 = Entry(tab2,width=50)
txt2.grid(column=1, row=0)
lbl2.grid(column=0, row=0)

btn3 = Button(tab2, text="Video Göster", command=showVideo)
btn3.grid(column=2, row=0)
btn4 = Button(tab2,height=5,width=20, text="Video Maske Tespit Et",compound='c', command=detectVideoMask)
btn4.grid(column=1, row=2)

btn5 = Button(tab3,height=10,width=20, text="Kamera Maske Tespit Et",compound='c', command=detectCamMask)
btn5.grid(column=1, row=2)

tab_control.pack(expand=1, fill='both')
window.mainloop()
