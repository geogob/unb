#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 00:33:06 2020

@author: george
"""


import numpy as np
import cv2
frame = None
mask = None
thr=40
def mouse_drawing(event, x, y, flags, params):
    global frame
    # global mask
    # frame = mask
    global thr
    if(event== cv2.EVENT_LBUTTONDOWN):
        (B, G, R) = cv2.split(frame)
        h,w = frame.shape
        valueB = B[y,x]
        valueG = G[y,x]
        valueR = R[y,x]
        c, l = B.shape
        for i in range(h):
            for j in range(w):
                if B[i,j]>=valueB-thr & B[i,j]<=valueB-thr & G[i,j]>=valueG-thr & G[i,j]<=valueG-thr & R[i,j]>=valueR-thr & R[i,j]<=valueR-thr:
                    frame[]=(0,0,255)
                    cv2.imshow("Frame", frame) #mostra o frame na tela
        print(valueB, valueG, valueR)

#vídeo da webcam
cap = cv2.VideoCapture(0)

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_drawing)#mouse event

#loop de leitura do vídeo
while(True):
    existeFrame, frame = cap.read() #em cada loop recebe um frame e guarda na variavel frame.  Existe frame recebe True se existe frame
  
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([0,0,0])
    upper_green = np.array([100, 100, 40])
    
    mask = cv2.inRange(hsv, lower_green, upper_green)
    #hsv= hsv[:,:,10]
    
    cv2.imshow("HSV", mask) #mostra o frame na tela
    cv2.imshow("Frame", frame) #mostra o frame na tela
    if(cv2.waitKey(5) & 0xFF ==ord('q')): #espera 1 ms para ir pro prox frame e encerra se pressionar Q
        break

cap.release()
cv2.destroyAllWindows()
    
    