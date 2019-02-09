#!/usr/bin/env python
# coding: utf-8
This code gives as output the slices of end-diastolic and end-systolic frames of 100 patients, including both the initial and segmented images.

To know how many slices (N) the patient number X has, use:
N = slice_count[X] - slice_count[X-1]

To get his slice number n (n being a number between 1 and N, including both), use:
Y = output[slice_count[X-1] + n-1]

Then:
Y[0] = patient number
Y[1] = slice number
Y[2] = ED 2D original image
Y[3] = ED 2D ground truth
Y[4] = ES 2D original image
Y[5] = ES 2D ground truth
# In[1]:


import os
import numpy as np
import SimpleITK as sitk


# In[2]:


l=[]
for i, name in enumerate(os.listdir('Data')):
    data = open('Data\{}\Info.cfg'.format(name), 'r')
    
    ED=data.readline()    #end-diastolic frame information
    for s in ED.split():
        if s.isdigit():   #end-diastolic frame number
            #reading the end-diastolic 3d images:
            if int(s)<10:
                im_EDframe= sitk.ReadImage('Data\{}\{}_frame0{}.nii.gz'.format(name,name,s))
                im_EDgt=sitk.ReadImage('Data\{}\{}_frame0{}.nii.gz'.format(name,name,s))
            else:
                im_EDframe= sitk.ReadImage('Data\{}\{}_frame{}.nii.gz'.format(name,name,s))
                im_EDgt=sitk.ReadImage('Data\{}\{}_frame{}_gt.nii.gz'.format(name,name,s))
                
    ES=data.readline()    #end-systolic frame information
    for s in ES.split():
        if s.isdigit():   #end-systolic frame number
            #reading the end-systolic 3d images:
            if int(s)<10:
                im_ESframe= sitk.ReadImage('Data\{}\{}_frame0{}.nii.gz'.format(name,name,s))
                im_ESgt=sitk.ReadImage('Data\{}\{}_frame0{}.nii.gz'.format(name,name,s))
            else:
                im_ESframe= sitk.ReadImage('Data\{}\{}_frame{}.nii.gz'.format(name,name,s))
                im_ESgt=sitk.ReadImage('Data\{}\{}_frame{}_gt.nii.gz'.format(name,name,s))
                
    #Converting the 3d images into 3 dimensional arrays:        
    arr_EDframe= sitk.GetArrayFromImage(im_EDframe)
    arr_EDgt= sitk.GetArrayFromImage(im_EDgt)
    arr_ESframe= sitk.GetArrayFromImage(im_ESframe)
    arr_ESgt= sitk.GetArrayFromImage(im_ESgt)
    
    NSlices=arr_EDframe.shape[0]
    
    #l=list with the patient number, the number of slices per frame, and the four 3D frames as arrays
    l.append([i+1, NSlices,arr_EDframe,arr_EDgt,arr_ESframe,arr_ESgt])
    
    
#output=list with the patient number, the slices number, and the four 2D slices as arrays
slice_count=[0]
output=[]
for i in range(100):
    n=l[i][1]
    for h in range(n):
        output.append([l[i][0],h+1,l[i][2][h],l[i][3][h],l[i][4][h],l[i][5][h]])
    slice_count.append(slice_count[i]+n)

