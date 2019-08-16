# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
from random import randint
import features
old_frame=0
cap=cv2.VideoCapture(0)
w=cap.get(3)
h=cap.get(4)
frameArea=h*w
areaTH=frameArea/400

#Lines
line_up=int(2*(h/10))
up_limit=int(1*(h/10))


line_up_color=(255,0,0)
pt3 =  [0, line_up]
pt4 =  [w, line_up]
pts_L2 = np.array([pt3,pt4], np.int32)
pts_L2 = pts_L2.reshape((-1,1,2))

pt5 =  [0, up_limit]
pt6 =  [w, up_limit]
pts_L3 = np.array([pt5,pt6], np.int32)
pts_L3 = pts_L3.reshape((-1,1,2))

#Background Subtractor
fgbg=cv2.createBackgroundSubtractorMOG2(detectShadows=False)

#Kernals
kernalOp = np.ones((3,3),np.uint8)
kernalOp2 = np.ones((5,5),np.uint8)
kernalCl = np.ones((11,11),np.uint)


font = cv2.FONT_HERSHEY_SIMPLEX

old_x = 0
old_y = 0
old_w = 0
old_h = 0
old_cx = 0
old_cy = 0

count = 0
insertingFlag=False
while(cap.isOpened()):
    ret,frame=cap.read()
    fgmask=fgbg.apply(frame)
    cv2.imshow('MOG2', fgmask)
    if ret==True:

        #Binarization
        ret,imBin=cv2.threshold(fgmask,200,255,cv2.THRESH_BINARY)

        #OPening i.e First Erode the dilate
        mask=cv2.morphologyEx(imBin,cv2.MORPH_OPEN,kernalOp)

        #Closing i.e First Dilate then Erode
        mask=cv2.morphologyEx(imBin,cv2.MORPH_CLOSE,kernalCl)
        
        #Find Contours
        contours0,hierarchy=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        if len(contours0) != 0:
            cnt = max(contours0, key = cv2.contourArea) # Area Max
            area=cv2.contourArea(cnt)
            if area>areaTH:
                ####Tracking######
                m=cv2.moments(cnt)
                cx=int(m['m10']/m['m00'])
                cy=int(m['m01']/m['m00'])
                x,y,w,h=cv2.boundingRect(cnt)
                new=True
                if cv2.waitKey(1)&0xff==ord('p'):
                    cv2.imwrite("frames/frame%d.jpg" % count, frame)
                    count += 1
                if y+h > up_limit:
                    if insertingFlag == True:
                        p=features.try_match(frame)
                        # cv2.imshow("match",p.matching_Compute())
                        if old_y+old_h > line_up and y+h <= line_up:
                            insertingFlag=False
                    elif insertingFlag == False:   
                        if old_y+old_h < line_up and y+h >= line_up: # going down
                            insertingFlag=True

                cv2.circle(frame,(cx,y+h),5,(0,0,255),-1)
                img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

                old_x = x
                old_y = y
                old_w = w
                old_h = h
                old_cx = cx
                old_cy = cy                
        else:
            insertingFlag=False 

        if insertingFlag == True:
            status='STATUS: Inserting object.'
        elif insertingFlag == False:
            status='STATUS: No object.'

        frame=cv2.polylines(frame,[pts_L2],False,line_up_color,thickness=2)
        frame=cv2.polylines(frame,[pts_L3],False,(255,255,255),thickness=1)
        cv2.putText(frame, status, (10, 90), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('Frame',frame)
        # TESTE SUBTRAÇÃO
        # sub = (frame - old_frame)
        # old_frame = frame
        # cv2.imshow('Subtracao Simples', sub)
    if cv2.waitKey(1)&0xff==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
