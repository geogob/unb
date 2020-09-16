#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 12:30:17 2020

@author: George Oliveira Barros

matrícula: 190052911

Refs: 
https://www.learnopencv.com/camera-calibration-using-opencv/
https://docs.opencv.org/master/d9/d0c/group__calib3d.html
https://stackoverflow.com/questions/17665912/findchessboardcorners-fails-for-calibration-image
"""


import numpy as np
import cv2
import glob

# Wait time to show calibration in 'ms'
WAIT_TIME = 100

# termination criteria for iterative algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#board dimension (points)
cbrow = 6
cbcol = 9

#box size (2cm)
square_size=0.020

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# IMPORTANT : Object points must be changed to get real physical distance.
objp = np.zeros((cbrow * cbcol, 3), np.float32)
objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)

objp = objp * square_size
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('/home/george/unb/computer vision projects/desafio/img/*.jpg')

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
print("Matriz de Rotação: \n")
print(rvecs)
print("Vetor de translação : \n")
print(tvecs)

# ---------- Saving the calibration -----------------
cv_file = cv2.FileStorage("test.yaml", cv2.FILE_STORAGE_WRITE)
cv_file.write("camera_matrix", mtx)
cv_file.write("dist_coeff", dist)

# note you *release* you don't close() a FileStorage object
cv_file.release()
