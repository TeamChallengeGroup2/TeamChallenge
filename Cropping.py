"""
Cropping the Region of interest (ROI)

Team Challenge (TU/e & UU)
Team 2
"""

import numpy as np
import cv2
import math

def cropROI(arrayin, slicenr):
    # This function crops an image given an aray with indices 
    # [slicenr, imageX, imageY] and the slicenr as average over this number of slices
    
    LVradius = 20 #the radius used in frst, which should be the radius of the LV in the top image
    cropdiam = 64 #the length in X and Y direction of the cropped image
    
    multi_mindist = []
    multi_circles = []
    sumdist1 = 0
    sumdist2 = 0
    
    for i in range(slicenr):
        topslice=arrayin[i,:,:]
        
        # Find the coordinates of the center of the image
        center = [topslice.shape[1]/2,topslice.shape[0]/2]        

        im = np.array((topslice/np.amax(topslice)) * 255, dtype = np.uint8)
    
        circles = cv2.HoughCircles(im, cv2.HOUGH_GRADIENT, 1, 70, param1=25, param2=20, minRadius=LVradius-14, maxRadius=LVradius+10)
        multi_circles.append(circles)

    # Function to calculate distance from coordinates of HoughCircles to center of the image
        def calculateDistance(x1,y1,x2,y2):  
            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
            return dist

    # Calculate the distances and add to list dist
        dist = []
        for j in range(circles.shape[1]):
            d = calculateDistance(circles[0,j,0],circles[0,j,1],center[0],center[1])
            dist.append(d)

        mindist = np.argmin(dist)         #find index of minimal distance
        multi_mindist.append(mindist)
        sumdist1 += int(circles[0,mindist,1])
        sumdist2 += int(circles[0,mindist,0])

    sumdist1 = int(sumdist1/(slicenr))
    sumdist2 = int(sumdist2/(slicenr))
    
    #find the coordinates around which to crop
    if cropdiam >= sumdist1:
        cropcoor_x1 = 0
    else:
        cropcoor_x1 = sumdist1-cropdiam
    cropcoor_x2 = cropcoor_x1+2*cropdiam
    if cropcoor_x2 >= topslice.shape[0]:
        cropcoor_x2 = topslice.shape[0]
        cropcoor_x1 = cropcoor_x2-2*cropdiam
        
    if cropdiam >= sumdist2:
        cropcoor_y1 = 0
    else:
        cropcoor_y1 = sumdist2-cropdiam
    cropcoor_y2 = cropcoor_y1 + 2*cropdiam
    if cropcoor_y2 >= topslice.shape[1]:
        cropcoor_y2 = topslice.shape[1]
        cropcoor_y1 = cropcoor_y2-2*cropdiam
    
    croppedim = im[cropcoor_x1:cropcoor_x2,cropcoor_y1:cropcoor_y2]
      
    return croppedim, cropcoor_x1, cropcoor_x2, cropcoor_y1, cropcoor_y2

def cropImage(data):
    # CROPPING
    data_cropped = [] # List with the patient number, the slices number, and the four 2D slices as arrays
    slice_count = []
    excluded = []
    
    for j in range(len(data)):
    
        # Extract the ED frame for each patient
        EDframe = data[j][2]
        
        # Crop only if HoughCircles is able to find a circle
        cropped_EDim, EDx1, EDx2, EDy1, EDy2 = cropROI(EDframe,4)
        
        if cropped_EDim.size:
            # Extract the slice number
            n = data[j][1]
            slice_count.append(n)
            
            # Extract and save the ED and ES slices and ground truth slices
            for h in range(n):
                EDslice = data[j][2][h]
                EDslicegt = data[j][3][h]
                ESslice = data[j][4][h]
                ESslicegt = data[j][5][h]
                
                # Save the data in lists
                data_cropped.append([data[j][0],h+1,EDslice[EDx1:EDx2, EDy1:EDy2],EDslicegt[EDx1:EDx2, EDy1:EDy2],ESslice[EDx1:EDx2, EDy1:EDy2],ESslicegt[EDx1:EDx2, EDy1:EDy2],data[j][6]])
                
        else:
            excluded.append(data[j][0])
                
    print('Images cropped')
    
    return data_cropped, excluded
    
