import cv2
from functions import *

vid = cv2.VideoCapture(0)

#CONSTANTS:
show_contours = True
  
while(True):

    ret, frame = vid.read()
    if ret:
        thresholded = thresh(frame)
        # [all_cnts,cnts] = findcontours(frame,180)
        # if show_contours:
        #     cv2.drawContours(frame,cnts,-1,(0,255,0), 14)
        cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()