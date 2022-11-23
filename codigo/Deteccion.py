import torch
import cv2
import numpy as np

#Leemos el modelo
model = torch.hub.load('ultralytics/yolov5', 'custom', path='model\RedNeuronalCarro.pt',device='cpu',force_reload=True)
#Tomamos la camara como entrada, puede cambiarse el 0 por la ruta del video
cap = cv2.VideoCapture(0)

while True:
    #Leemos el frame de la camara
    ret,frame = cap.read()

    #Detecciones por frame
    detect = model(frame)

    #Dibujamos los rectangulos de las detecciones
    cv2.imshow('Detector de Carros',np.squeeze(detect.render()))

    #Si se presiona la tecla q se cierra el programa
    t = cv2.waitKey(1)
    if t == 27:
        break

#Cerramos la camara y la ventana
cap.release()
cv2.destroyAllWindows()

