import cv2
import numpy as np
from matplotlib import pyplot as plt

        
def retify():
    #CASAMENTO DE PONTOS COM SIFT
    img_path1 = '/home/george/unb/computer vision projects/trabalho 01/disparity/left-frame.jpg'
    img_path2 = '/home/george/unb/computer vision projects/trabalho 01/disparity/right-frame.jpg'
    img1= cv2.imread(img_path1)
    img2= cv2.imread(img_path2)
    
    h,w = img2.shape[:2]
    imageSize = img2.shape
    #imgSize = img1.shape
    #Recalibração das cameras
    # points1 = np.zeros((4,2))
    # points2 = np.zeros((4,2))
    # points1[0,:] = [105,22]; points1[1,:] = [297,17]; points1[2,:] = [22,317]; points1[3,:] = [510,229]
    # points2[0,:] = [333,29]; points1[1,:] = [506,50]; points1[2,:] = [98,186]; points1[3,:] = [454,308]
    caminho1 = '/home/george/unb/computer vision projects/trabalho 01/Calibration1 resize/*.jpg'
    ret1, mtx1, dist1, rvecs1, tvecs1, objpoints1, imgpoints1 = calibracao("camera1-resize.yaml", caminho1)
    
    caminho2 = '/home/george/unb/computer vision projects/trabalho 01/Calibration 2 resize/*.jpg'
    ret2, mtx2, dist2, rvecs2, tvecs2, objpoints2, imgpoints2 = calibracao("camera2-resize", caminho2)
    
    
    #Get optimal camera matrix for better undistortion 
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx1,dist1,(w,h),1,(w,h))
    new_camera_matrix2, roi = cv2.getOptimalNewCameraMatrix(mtx2,dist2,(w,h),1,(w,h))

    img_1_undistorted = cv2.undistort(img1, mtx1, dist1, None, new_camera_matrix)
    img_2_undistorted = cv2.undistort(img2, mtx2, dist2, None, new_camera_matrix2)
    
    #R, T, E, F = cv2.stereoCalibrate(objpoints1, imgpoints1, imgpoints2, mtx1, dist1, mtx2, dist2, imageSize)
    #Downsample each image 3 times (because they're too big)
    #img_1_downsampled = downsample_image(img_1_undistorted,3)
    #img_2_downsampled = downsample_image(img_2_undistorted,3)
    
    #Set disparity parameters
    #Note: disparity range is tuned according to specific parameters obtained through trial and error. 
    win_size = 5
    min_disp = -1
    max_disp = 63 #min_disp * 9
    num_disp = max_disp - min_disp # Needs to be divisible by 16
    
    #Create Block matching object. 
    stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
                                   numDisparities = num_disp,
                                   blockSize = 5,
                                   uniquenessRatio = 5,
                                   speckleWindowSize = 5,
                                   speckleRange = 5,
                                   disp12MaxDiff = 2,
                                   P1 = 8*3*win_size**2,#8*3*win_size**2,
                                   P2 =32*3*win_size**2) #32*3*win_size**2)

    #Compute disparity map
    print ("\nComputing the disparity  map...")
    disparity_map = stereo.compute(img_2_undistorted, img_1_undistorted)
    
    #Show disparity map before generating 3D cloud to verify that point cloud will be usable. 
    plt.imshow(disparity_map,'gray')
    plt.show()

    
    #Load focal length. 
    
    #focal_length = np.array(mtx1[0][0])

retify()
    