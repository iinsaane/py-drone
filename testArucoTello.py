import cv2
from cv2 import aruco
import numpy as np
import imutils
import os
import sys
import time
import math
from djitellopy import Tello

# set the aruco marker dictionary
marker_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
param_markers = aruco.DetectorParameters_create()

markerSize = 10 #cm

camera_matrix = np.array([[921.170702, 0.000000, 459.904354], [0.000000, 919.018377, 351.238301], [0.000000, 0.000000, 1.000000]])
distortion = np.array([-0.033458, 0.105152, 0.001256, -0.006647, 0.000000])

def normalize(value, minimum, maximum):
    if (value < minimum):
        value = minimum
    if (value > maximum):
        value = maximum
    if (value == 0):
        value = 0
    return value

def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def withinRadius(x1, y1, x2, y2, radius):
    return distance(x1, y1, x2, y2) < radius

    

# initialize drone
print("Initializing drone...")
tello = Tello()
print("Connecting...")
try:
    tello.connect()
except:
    pass


# get opencv video stream from tello
tello.streamon()
frame_read = tello.get_frame_read()
lastPos = (0, 0)
marker_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
param_markers = aruco.DetectorParameters_create()
while True:
    frame = frame_read.frame
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_ids, rejected_candidates = aruco.detectMarkers(gray, marker_dict, parameters=param_markers)
    # frame_aruco = aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

    
    if marker_corners:
        for id, corner in zip(marker_ids, marker_corners):
            if(id == 0) :

                leftTopCorner = (
                    math.floor(corner[0][0][0]), 
                    math.floor(corner[0][0][1])
                )
                rightTopCorner = (
                    math.floor(corner[0][1][0]), 
                    math.floor(corner[0][1][1])
                )
                leftBottomCorner = (
                    math.floor(corner[0][2][0]), 
                    math.floor(corner[0][2][1])
                )
                rightBottomCorner = (
                    math.floor(corner[0][3][0]), 
                    math.floor(corner[0][3][1])
                )

                # estimate distance to marker
                rvec , tvec, _ = aruco.estimatePoseSingleMarkers(corner, markerSize, camera_matrix, distortion)
                # print out the distance to the marker
                try:
                    print(rvec)
                except:
                    pass

                #get center of marker
                cx = int(np.mean(corner[:,:,0]))
                cy = int(np.mean(corner[:,:,1]))
                
                #draw an arrow from center of the screen to center of marker
                cv2.arrowedLine(frame, (320, 240), (cx, cy), (0, 0, 255), 2)

                #check if marker is not within radius of the center of the screen
                # if(not withinRadius(320, 240, cx, cy, 100)):
                #     print("not within radius")
                    


                lastPos = (cx, cy)
    else: 
        if(lastPos):
            cv2.arrowedLine(frame, (320, 240), lastPos, (0, 0, 255), 2)
            lastPos = None


    # show frame
    cv2.imshow('frame', frame)

    time.sleep(1/30)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        tello.streamoff
        break

tello.land()
