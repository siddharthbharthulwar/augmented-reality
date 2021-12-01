import cv2 as cv
import numpy as np
from functions import *
from cube import *
import matplotlib.pyplot as plt
import math

K=np.array([[1406.08415449821,0,0],
           [2.20679787308599, 1417.99930662800,0],
           [1014.13643417416, 566.347754321696,1]])

cap = cv.VideoCapture("dynamic.mp4")

while(cap.isOpened()):
    ret, frame = cap.read()

    # frame = cv.imread("static.png", cv.IMREAD_COLOR)
    [all_cnts, cnts] = findcontours(frame, 180)
    cv.drawContours(frame, cnts, -1, (0, 255, 0), 14)
    [tag_cnts, corners] = approx_quad(cnts)
    cv2.drawContours(frame,tag_cnts,-1,(255,0,0), 4)

    for i,tag in enumerate(corners):
        # find number of points in the polygon
        num_points = num_points_in_poly(frame,tag_cnts[i])

        # set the dimension for homography
        dim = int(math.sqrt(num_points))

        H = homography(tag,dim)
        H_inv = np.linalg.inv(H)

                # get squared tag
        square_img = warp(H_inv,frame,dim,dim)

            # threshold the squared tag
        imgray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
        ret, square_img = cv2.threshold(imgray, 180, 255, cv2.THRESH_BINARY)

            #decode squared tile
        [tag_img,id_str,orientation] = encode_tag(square_img)

        ##########cube mode#################

        
        # Find top corners for the cube  
        H=homography(tag,200)
        H_inv = np.linalg.inv(H)
        P=projection_mat(K,H_inv)
        new_corners=cubePoints(tag, H, P, 200)

        # draw the cube onto the frame
        frame=drawCube(tag, new_corners,frame,(0, 255, 0),(255, 0, 0),False)

        ##########cube mode##################
        cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break