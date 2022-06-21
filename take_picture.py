import cv2

cap = cv2.VideoCapture(2) # video capture source camera (Here webcam of laptop) 
i = 0
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
while(True):
    ret,frame = cap.read() # return a single frame in variable `frame`
    cv2.imshow('img',frame) #display the captured image
    if cv2.waitKey(1) & 0xFF == ord('q'): #save on pressing 'y' 
        cv2.imwrite('calibration/c' + str(i) + '.png',frame)
        cv2.destroyAllWindows()
        i= i+1

cap.release()