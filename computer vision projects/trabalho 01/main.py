#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 12:30:17 2020

@author: George Oliveira Barros 

"""

import numpy as np
import cv2
import glob

#REQUISITO 1: Realize a calibração dos intrínsecos das duas câmeras usadas para capturar as imagens deste trabalho, usando as imagens disponíveis nos diretórios Calibration1 e Calibration2.

def calibracao(tituloSave, caminho):
    # Wait time to show calibration in 'ms'
    WAIT_TIME = 100
    
    # termination criteria for iterative algorithm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # generalizable checkerboard dimensions
    # https://stackoverflow.com/questions/31249037/calibrating-webcam-using-python-and-opencv-error?rq=1
    cbrow = 8
    cbcol = 6
    square_size=0.020
    
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
        print(fname)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (cbrow,cbcol), None)
        #print(ret)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
    
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
            
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (cbrow,cbcol ), corners2,ret)
            cv2.imshow('img', img)
            cv2.waitKey(WAIT_TIME)
    
    cv2.destroyAllWindows()
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    
    print("Matriz de íntrínsecos : \n")
    print(mtx)
    print("Coeficientes de distorção da lente: \n")
    print(dist)
    print("Vetor de Rotação: \n")
    print(rvecs)
    print("Vetor de translação : \n")
    print(tvecs)

    # ---------- Saving the calibration -----------------
    cv_file = cv2.FileStorage(tituloSave, cv2.FILE_STORAGE_WRITE)
    cv_file.write("camera_matrix", mtx)
    cv_file.write("dist_coeff", dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()

print("CALIBRANDO CAMERA 1...")
#calibração da câmera 1
caminho1 = '/home/george/unb/computer vision projects/trabalho 01/trabalho1_imagens/Calibration1/*.jpg'
calibracao("camera1.yaml", caminho1)

print("CALIBRANDO CAMERA 2...")
#calibração da câmera 2
caminho2 = '/home/george/unb/computer vision projects/trabalho 01/trabalho1_imagens/Calibration2/*.jpg'
calibracao("camera2.yaml", caminho2)


#REQUISITO 2: Estimativa da pose das câmeras
def estimacaoPose():
    #aqui
    print("estimacao da pose das cameras")

estimacaoPose()