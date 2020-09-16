#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 13:31:21 2020

@author: george
"""

import cv2

img = cv2.imread('img/my_photo-4.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imshow('ex', gray)

cv2.destroyAllWindows()

ret, corners = cv2.findChessboardCorners(gray, (6,9), None)