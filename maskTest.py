import cv2
import numpy as np
import imutils
import os
import sys
import time

def remove_isolated_pixels(image):
    connectivity = 8

    output = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_32S)

    num_stats = output[0]
    labels = output[1]
    stats = output[2]

    new_image = image.copy()

    for label in range(num_stats):
        if stats[label,cv2.CC_STAT_AREA] == 1:
            new_image[labels == label] = 0

    return new_image

# Creating kernel
kernel = np.ones((5, 5), np.uint8)

# take frames from the camera and show them on screen
cam = cv2.VideoCapture(2)

lasttime = time.time()
while True:
    ret, frame = cam.read()
    canvas = np.zeros(frame.shape, np.uint8)
    #process frames

    #convert image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (50, 100, 50), (80, 255,255))
    mask = remove_isolated_pixels(mask)
    mask = cv2.erode(mask, kernel) 

    #find the countours in the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # get the contour with the largest area
    max_area = 0
    largest_contour = ""
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            largest_contour = cnt

    # find the center of the contour
    if largest_contour != "":
        M = cv2.moments(largest_contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.circle(canvas, (cx, cy), 5, (0, 0, 255), -1)

    # straighten the countour while keeping its shape and orientation
    if largest_contour != "":
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(canvas, [box], 0, (0, 255, 0), 2)

    

    #draw the largest contour on a black binary canvas
    if largest_contour != "":
        cv2.drawContours(frame, [largest_contour], -1, (255, 255, 255), 3)

    # put mask and image next to each other in the same
    cv2.imshow("canvas", canvas)
    cv2.imshow('video', frame)
    
    #get fps and print it
    # print("FPS: ", 1.0 / (time.time() - lasttime))
    lasttime = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break