#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 12:30:17 2020

@author: George Oliveira Barros 

"""

import numpy as np
import cv2
import glob

#REQUISITO 1: 
def calibracao(tituloSave, caminho):
    print("Requisito 1 \n")
    # Wait time to show calibration in 'ms'
    WAIT_TIME = 100
    
    # termination criteria for iterative algorithm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # generalizable checkerboard dimensions
    # https://stackoverflow.com/questions/31249037/calibrating-webcam-using-python-and-opencv-error?rq=1
    cbrow = 8
    cbcol = 6
    square_size=0.025 #aproximadamente 0.24 x 0.25
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # IMPORTANT : Object points must be changed to get real physical distance.
    objp = np.zeros((cbrow * cbcol, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)
    
    objp = objp * square_size
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    images = glob.glob(caminho)
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #print(fname)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (cbrow,cbcol), None)
        #print(ret)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
    
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
            
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (cbrow, cbcol), corners2,ret)
            cv2.imshow('img', img)
            cv2.waitKey(WAIT_TIME)
    
    cv2.destroyAllWindows()
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    
    print("Matriz de íntrínsecos : \n")
    print(mtx)
    print("Coeficientes de distorção da lente: \n")
    print(dist)
    # print("Vetor de Rotação: \n")
    # print(rvecs)
    # print("Vetor de translação : \n")
    # print(tvecs)

    # ---------- Saving the calibration -----------------
    cv_file = cv2.FileStorage(tituloSave, cv2.FILE_STORAGE_WRITE)
    cv_file.write("camera_matrix", mtx)
    cv_file.write("dist_coeff", dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()
    
    return mtx, dist

print("CALIBRANDO CAMERA 1...")
#calibração da câmera 1
caminho1 = '/home/george/unb/computer vision projects/trabalho 01/trabalho1_imagens/Calibration1/*.jpg'
mtx1, dist1 = calibracao("camera1.yaml", caminho1)

print("CALIBRANDO CAMERA 2...")
#calibração da câmera 2
caminho2 = '/home/george/unb/computer vision projects/trabalho 01/trabalho1_imagens/Calibration2/*.jpg'
mtx2, dist2 = calibracao("camera2.yaml", caminho2)


#REQUISITO 2: 
print("Requisito 2...")
def estimacaoPose(mtx, dist, objP, imgP):
    #aqui
    #dist=np.array([1,1,1,1,1])
    # Find the rotation and translation vectors.
    #rvec, tvec, retval = cv2.solvePnPRansac(objP, imgP, mtx, dist)
    retval, rvec, tvec = cv2.solvePnP(objP, imgP, mtx, dist) 
    return retval, rvec, tvec


#localização dos objetos nas coordendas do mundo
a=[0, 0, 0]
b=[0, 1.4, 0]
c=[2.6, 0, 0]
d=[2.6, 1.4, 0]
objP_mundo = np.zeros((4,3), np.float32)

objP_mundo[0,:]=a
objP_mundo[1,:]=b
objP_mundo[2,:]=c
objP_mundo[3,:]=d

print("-------------------")
#localização 2d dos objetos (no frame do vídeo 1) na CAMERA 1
aCam1=[1066,92]
bCam1=[1615,164]
cCam1=[314,594]
dCam1=[1451,982]
#deixando no formato que a função precisa
imgP_cam1 = np.zeros((4,2), np.float32)
imgP_cam1[0,:]=aCam1
imgP_cam1[1,:]=bCam1
imgP_cam1[2,:]=cCam1
imgP_cam1[3,:]=dCam1
#imgP_cam1 = [aCam1, bCam1, cCam1, dCam1]

#pose estimation from camera 1
retval1, rvec1, tvec1 = estimacaoPose(mtx1, dist1, objP_mundo, imgP_cam1)
print("Extrínsecos da camera 1...\n")
print("Vetor de rotacao:\n")
print(rvec1)
print("Vetor de translação:\n")
print(tvec1)
rotM1 = cv2.Rodrigues(rvec1)[0]
cameraPosition1 = -np.matrix(rotM1).T * np.matrix(tvec1)
print("Camera 1 Position:\n")
print(cameraPosition1) 


print("-------------------")
#localização 2d dos objetos (no frame do vídeo 2) na CAMERA 2
aCam2=[219,48]
bCam2=[626,31]
cCam2=[50,675]
dCam2=[1084,485]
#deixando no formato que a função precisa
imgP_cam2 = np.zeros((4,2), np.float32)
imgP_cam2[0,:]=aCam2
imgP_cam2[1,:]=bCam2
imgP_cam2[2,:]=cCam2
imgP_cam2[3,:]=dCam2
#pose estimation from camera 2
retval2, rvec2, tvec2 = estimacaoPose(mtx2, dist2, objP_mundo, imgP_cam2)
print("Extrínsecos da camera 2...\n")
print("Vetor de rotação:\n")
print(rvec2)
print("Vetor de translação:\n")
print(tvec2)
rotM2 = cv2.Rodrigues(rvec2)[0]
cameraPosition2 = -np.matrix(rotM2).T * np.matrix(tvec2)
print("Camera 2 Position:\n")
print(cameraPosition2)

#REQUISITO 3: 
def mapas():
    #aqui
    print("Requisito 3")

mapas()


#REQUISITO 4: 
def rastreamento():
    #aqui
    print("Requisito 4")

rastreamento()