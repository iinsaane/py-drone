import cv2 #4.5.5
from cv2 import aruco #4.5.5
import numpy as np
import imutils
import os
import sys
import time
import math
from inspect import getmembers, isfunction

def rad2deg(rad):
    return rad * 180 / math.pi

# rotation matrix to euler angle
# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

# set the aruco marker dictionary
marker_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
param_markers = aruco.DetectorParameters_create()

#tracker info
idToTrack = 0
markerSize = 10 #cm

#camera info
# [[1.67704400e+03 0.00000000e+00 9.58665983e+02]
#  [0.00000000e+00 1.68882316e+03 4.74350488e+02]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
camera_matrix = np.array([[1677.04, 0.0, 958.666], [0.0, 1688.82, 474.35], [0.0, 0.0, 1.0]])

# camera_matrix = np.array( [ [1.63372397, 0.00000000, 9.20161723], [0.00000000, 1.63980861, 5.41642560], [0.00000000, 0.00000000, 1.00000000] ] )
# [[-0.09406613  1.43386179 -0.01567281  0.00815202 -3.80767903]]
distortion = np.array([[-0.09406613, 1.43386179, -0.01567281, 0.00815202, -3.80767903]])

# 180 deg rotation matrix
R_flip = np.zeros((3,3), dtype=np.float32)
R_flip[0,0] = 1.0
R_flip[1,1] =-1.0
R_flip[2,2] =-1.0

# text info
font = cv2.FONT_HERSHEY_SIMPLEX

# get camera stream (1080p)
cam = cv2.VideoCapture(2)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

lastPos = (0, 0)

# loop frames
while True:
    # get frame
    ret, frame = cam.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_ids, rejected_candidates = aruco.detectMarkers(gray, marker_dict, parameters=param_markers)
    frame_aruco = aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

    
    if marker_corners:
        for id, corner in zip(marker_ids, marker_corners):
            if(id == idToTrack) :
                
                 # estimate distance to marker
                ret = aruco.estimatePoseSingleMarkers(corner, markerSize, camera_matrix, distortion)
                rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]
                
                aruco.drawAxis(frame, camera_matrix, distortion, rvec, tvec, 10)

                str_position = "MARKER Position x=%4.2f y=%4.2f z=%4.2f" % (tvec[0], tvec[1], tvec[2])
                cv2.putText(frame, str_position, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                # obtian the rotation matrix tag -> camera
                R_ct = np.matrix(cv2.Rodrigues(rvec)[0])
                R_tc = R_ct.T

                # get the attitude in terms of euler 321
                roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(R_flip*R_tc)
                
                # print the markers attitude respect to camera frame
                strAttitude = "MARKER Attitude r=%4.0f p=%4.0f y=%4.0f" % (math.degrees(roll_marker), math.degrees(pitch_marker), math.degrees(yaw_marker))
                cv2.putText(frame_aruco, strAttitude, (10, 90), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

                # get the Position and attitude of the camera respect to the marker frame
                # print(np.matrix(tvec))
                pos_camera = np.matrix(tvec) * -R_tc
                # pos_camera = pos_camera.T
                pos_camera = np.array(pos_camera)[0]
                
                
                roll_camera, pitch_camera, yaw_camera = rotationMatrixToEulerAngles(R_flip*R_ct)

                strPosCam = "CAMERA Position x=%4.0f y=%4.0f z=%4.0f" % (pos_camera[0], pos_camera[1], pos_camera[2])
                cv2.putText(frame_aruco, strPosCam, (10, 150), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

                strAttCam = "CAMERA Attitude r=%4.0f p=%4.0f y=%4.0f" % (math.degrees(roll_camera), math.degrees(pitch_camera), math.degrees(yaw_camera))
                cv2.putText(frame_aruco, strAttCam, (10, 210), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                

    # shrink the frame to half its size
    frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

    # show the frame
    cv2.imshow('frame', frame)

    # quit on q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break