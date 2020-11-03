#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 23:44:31 2020

@author: george
"""

import cv2
 
# Opens the Video file
cap= cv2.VideoCapture('/home/george/unb/computer vision projects/trabalho 01/trabalho1_imagens/camera2.webm')
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite('kang'+str(i)+'.jpg',frame)
    i+=1
 
cap.release()
cv2.destroyAllWindows()