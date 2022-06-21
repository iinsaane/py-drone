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
    frame = cv2.GaussianBlur(frame, (7, 7), 0)

    #convert image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (50, 70, 50), (80, 255,255))
    mask = remove_isolated_pixels(mask)
    mask = cv2.erode(mask, kernel)

    canny = cv2.Canny(mask, 50, 150, 255)

    # find the countours in the mask
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
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
    
        


    cv2.imshow("mask", mask)
    cv2.imshow("edged", canny)
    cv2.imshow("frame", frame)

    
    lasttime = time.time()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break