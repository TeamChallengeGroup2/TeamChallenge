#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
from keras import backend as K
import cv2

import SimpleITK as sitk

K.set_image_data_format('channels_last')  # TF dimension ordering in this code


# In[ ]:


#This function resizes the images so all of them have the same spacing

def respace(itk_image, new_spacing):
    spacing = itk_image.GetSpacing()
    size = itk_image.GetSize()
    new_size = (np.round(size*(spacing/np.array(new_spacing)))).astype(int).tolist()
    new_image = sitk.Resample(itk_image, new_size, sitk.Transform(),sitk.sitkNearestNeighbor,
                            itk_image.GetOrigin(), new_spacing, itk_image.GetDirection(), 0.0,
                            itk_image.GetPixelID())
    return new_image

#This function performs Adaptative Histogram Equalization on the images
#(Visualize patient 21 to see why it can be useful)

def equalize(itk_image, a, b):
    X=sitk.AdaptiveHistogramEqualizationImageFilter()
    X.SetAlpha(a)
    X.SetBeta(b)
    new_image = X.Execute(itk_image)
    return new_image
    


# In[ ]:


#Parameters that may improve performance when tuning:
    #The range of slices it is averaging (you would have to change the code where the comment is #**)
    #The radius of the circles we are looking for
    #The parameters inside the cv2.HoughCircles function

def cropROI(arrayin, slicenr):
    #input: an array with indices [slicenr, imageX, imageY]
    # slicenr: averaging over this number of slices
    
    LVradius=20 #the radius used in frst, which should be the radius of the LV in the top image
    cropdiam=64 #the length in X and Y direction of the cropped image
    multi_mindist = []
    multi_circles = []
    sumdist1 = 0
    sumdist2 = 0
    
    for i in range(slicenr): #**
        topslice=arrayin[i,:,:]
        center=[topslice.shape[1]/2,topslice.shape[0]/2]        #find coordinates of center of image

        im = np.array((topslice/np.amax(topslice)) * 255, dtype = np.uint8)
    
        circles=cv2.HoughCircles(im, cv2.HOUGH_GRADIENT, 1, 70, param1=25, param2=20, minRadius=LVradius-14, maxRadius=LVradius+10)
        multi_circles.append(circles)

    #function to calculate distance from coordinates of HoughCircles to center of the image
        def calculateDistance(x1,y1,x2,y2):  
            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
            return dist

    #calculate distances and add to list dist
        dist=[]
        for j in range(circles.shape[1]):
            d=calculateDistance(circles[0,j,0],circles[0,j,1],center[0],center[1])
            dist.append(d)

        mindist=np.argmin(dist)         #find index of minimal distance
        multi_mindist.append(mindist)
        sumdist1 += int(circles[0,mindist,1])
        sumdist2 += int(circles[0,mindist,0])

    sumdist1 = int(sumdist1/slicenr) #**
    sumdist2 = int(sumdist2/slicenr) #**
    
    #find the coordinates around which to crop
    if cropdiam>=sumdist1:
        cropcoor_x1=0
    else:
        cropcoor_x1=sumdist1-cropdiam
    cropcoor_x2=cropcoor_x1+2*cropdiam
    if cropcoor_x2>=topslice.shape[0]:
        cropcoor_x2=topslice.shape[0]
        cropcoor_x1=cropcoor_x2-2*cropdiam
        
    if cropdiam>=sumdist2:
        cropcoor_y1=0
    else:
        cropcoor_y1=sumdist2-cropdiam
    cropcoor_y2=cropcoor_y1+2*cropdiam
    if cropcoor_y2>=topslice.shape[1]:
        cropcoor_y2=topslice.shape[1]
        cropcoor_y1=cropcoor_y2-2*cropdiam
    
      
    return cropcoor_x1, cropcoor_x2, cropcoor_y1, cropcoor_y2


# In[ ]:


#This function gives as output a list with the end-diastolic and end-systolic 3D ground truths of 100 patients, and a list with
#the cropped 3D ground truths. If the output is the lists Y,X:
#Y[0] = X[0] = patient number
#Y[1] = X[1] = number of slices per frame
#Y[2] = ED 3D ground truth array; X[2] = ED 3D ground truth CROPPED array
#Y[3] = ES 3D ground truth array; X[3] = ES 3D ground truth CROPPED array

def load_crop(path):
    #"path" is the path of the folder named "Data"
    gt=[]
    cropgt=[]
    for i, name in enumerate(os.listdir(path)):
        data = open('{}\{}\Info.cfg'.format(path,name), 'r')
    
        ED=data.readline()    #end-diastolic frame information
        for s in ED.split():
            if s.isdigit():   #end-diastolic frame number
            #reading the end-diastolic 3d images:
                if int(s)<=9:
                    im_EDframe= sitk.ReadImage('{}\{}\{}_frame0{}.nii.gz'.format(path,name,name,s))
                    im_EDgt=sitk.ReadImage('{}\{}\{}_frame0{}_gt.nii.gz'.format(path,name,name,s))
                else:
                    im_EDframe= sitk.ReadImage('{}\{}\{}_frame{}.nii.gz'.format(path,name,name,s))
                    im_EDgt=sitk.ReadImage('{}\{}\{}_frame{}_gt.nii.gz'.format(path,name,name,s))
                
        ES=data.readline()    #end-systolic frame information
        for s in ES.split():
            if s.isdigit():   #end-systolic frame number
                #reading the end-systolic 3d images:
                if int(s)<=9:
                    im_ESframe= sitk.ReadImage('{}\{}\{}_frame0{}.nii.gz'.format(path,name,name,s))
                    im_ESgt=sitk.ReadImage('{}\{}\{}_frame0{}_gt.nii.gz'.format(path,name,name,s))
                else:
                    im_ESframe= sitk.ReadImage('{}\{}\{}_frame{}.nii.gz'.format(path,name,name,s))
                    im_ESgt=sitk.ReadImage('{}\{}\{}_frame{}_gt.nii.gz'.format(path,name,name,s))
                    
        
        #Define the new spacing:
        z=im_EDframe.GetSpacing()[2]
        new_spacing=(1.0,1.0,z)
        
        if im_EDframe.GetSpacing()!=im_EDgt.GetSpacing() or im_ESframe.GetSpacing()!=im_ESgt.GetSpacing():
            im_EDgt.SetSpacing(im_EDframe.GetSpacing())
            im_ESgt.SetSpacing(im_ESframe.GetSpacing())
        
        #Apply the same spacing in the X and Y dimensions to the images
        im_EDframe=respace(im_EDframe,new_spacing)
        im_EDgt=respace(im_EDgt,new_spacing)
        im_ESframe=respace(im_ESframe,new_spacing)
        im_ESgt=respace(im_ESgt,new_spacing)
        
        
        #Perform histogram equalization on the images
        #im_EDframe=equalize(im_EDframe, 0.5, 0.5)?
        #im_ESframe=equalize(im_ESframe, 0.5, 0.5)?
        
        #Here you can add other functions:
        
        
        #Converting the 3d images into 3 dimensional arrays:        
        arr_EDframe= sitk.GetArrayFromImage(im_EDframe)
        arr_EDgt= sitk.GetArrayFromImage(im_EDgt)
        arr_ESframe= sitk.GetArrayFromImage(im_ESframe)
        arr_ESgt= sitk.GetArrayFromImage(im_ESgt)
        
    
        NSlices=arr_EDframe.shape[0]
        
        #Crop the images
        #Tune av_n to improve cropping
        av_n=int(NSlices*3/4)
        
        EDx1, EDx2, EDy1, EDy2=cropROI(arr_EDframe,av_n)
        ESx1, ESx2, ESy1, ESy2=cropROI(arr_ESframe,av_n)

        cropEDgt=arr_EDgt[:,EDx1:EDx2, EDy1:EDy2]
        cropESgt=arr_ESgt[:,EDx1:EDx2, EDy1:EDy2]
            
     
        #gt=list with the patient number, the number of slices per frame, and the two 3D ground truths as arrays
        gt.append([i+1, NSlices, arr_EDgt, arr_ESgt])
        
        #cropgt= same as gt but the frames are cropped
        cropgt.append([i+1, NSlices, cropEDgt, cropESgt])
        
    return gt,cropgt


# In[ ]:


#Function to evaluate how much of the labelled region is inside the cropped image

def evaluate_crop(gt_list, cropgt_list):
    
    for i in range(100):
        gt=gt_list[i][2]
        gt[gt<=2.]=0
        gtn=gt.sum()
        
        cropgt=cropgt_list[i][2]
        cropgt[cropgt<=2.]=0
        cropn=cropgt.sum()
        
        if gtn!=0:
            h=(cropn/gtn)*100
            print('ED patient ',i+1,' = ',h,' % cropped')
        else: 
            print('no label in ED frame for patient ',i+1)
            
        gt=gt_list[i][3]
        gt[gt<=2.]=0
        gtn=gt.sum()
        
        cropgt=cropgt_list[i][3]
        cropgt[cropgt<=2.]=0
        cropn=cropgt.sum()
        
        if gtn!=0:
            h=(cropn/gtn)*100
            print('ES patient ',i+1,' = ',h,' % cropped')
        else: 
            print('no label in ES frame for patient ',i+1)    


# In[ ]:


#Define the path to the folder 'Data'
datapath=

gt_list, cropgt_list =load_crop(datapath)

evaluate_crop(gt_list, cropgt_list)

