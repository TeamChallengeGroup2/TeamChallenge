# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:44:10 2019

Team Challenge (TU/e & UU)
Team 2
"""

import numpy as np
import cv2
import math

def cropROI(arrayin, slicenr):
    # This function crops an image given an aray with indices 
    # [slicenr, imageX, imageY] and the slicenr as average over this number of slices
    
    LVradius=20 #the radius used in frst, which should be the radius of the LV in the top image
    cropdiam=63 #the length in X and Y direction of the cropped image
    
    multi_mindist = []
    multi_circles = []
    sumdist1 = 0
    sumdist2 = 0
    
    for i in range(slicenr):
        topslice=arrayin[i,:,:]
        
        # Find the coordinates of the center of the image
        center=[topslice.shape[1]/2,topslice.shape[0]/2]        

        im = np.array((topslice/np.amax(topslice)) * 255, dtype = np.uint8)
    
        circles=cv2.HoughCircles(im, cv2.HOUGH_GRADIENT, 1, 70, param1=25, param2=20, minRadius=LVradius-14, maxRadius=LVradius+10)
        multi_circles.append(circles)

    # Function to calculate distance from coordinates of HoughCircles to center of the image
        def calculateDistance(x1,y1,x2,y2):  
            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
            return dist

    # Calculate the distances and add to list dist
        dist=[]
        for j in range(circles.shape[1]):
            d=calculateDistance(circles[0,j,0],circles[0,j,1],center[0],center[1])
            dist.append(d)

        mindist=np.argmin(dist)         #find index of minimal distance
        multi_mindist.append(mindist)
        sumdist1 += int(circles[0,mindist,1])
        sumdist2 += int(circles[0,mindist,0])

    sumdist1 = int(sumdist1/(slicenr))
    sumdist2 = int(sumdist2/(slicenr))
    
    # Find the coordinates around which to crop
    cropcoor_x1=sumdist1-cropdiam
    cropcoor_x2=sumdist1+cropdiam
    cropcoor_y1=sumdist2-cropdiam
    cropcoor_y2=sumdist2+cropdiam
    
    croppedim=im[sumdist1-cropdiam:sumdist1+cropdiam,sumdist2-cropdiam:sumdist2+cropdiam]
      
    return croppedim, cropcoor_x1, cropcoor_x2, cropcoor_y1, cropcoor_y2
