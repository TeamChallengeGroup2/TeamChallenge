#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import PIL
import keras


# In[ ]:


#This function gives as output the end-diastolic and end-systolic 3D frames of 100 patients, including both the initial 
#and segmented images. If the output is the list Y:
#Y[0] = patient number
#Y[1] = number of slices per frame
#Y[2] = ED 3D original image AS AN ARRAY
#Y[3] = ED 3D ground truth AS AN ARRAY
#Y[4] = ES 3D original image AS AN ARRAY
#Y[5] = ES 3D ground truth AS AN ARRAY

def load3Ddata(path):
    #"path" is the path of the folder named "Data"
    l=[]
    for i, name in enumerate(os.listdir(path)):
        data = open('{}\{}\Info.cfg'.format(path,name), 'r')
    
        ED=data.readline()    #end-diastolic frame information
        for s in ED.split():
            if s.isdigit():   #end-diastolic frame number
            #reading the end-diastolic 3d images:
                if int(s)<10:
                    im_EDframe= sitk.ReadImage('{}\{}\{}_frame0{}.nii.gz'.format(path,name,name,s))
                    im_EDgt=sitk.ReadImage('{}\{}\{}_frame0{}.nii.gz'.format(path,name,name,s))
                else:
                    im_EDframe= sitk.ReadImage('{}\{}\{}_frame{}.nii.gz'.format(path,name,name,s))
                    im_EDgt=sitk.ReadImage('{}\{}\{}_frame{}_gt.nii.gz'.format(path,name,name,s))
                
        ES=data.readline()    #end-systolic frame information
        for s in ES.split():
            if s.isdigit():   #end-systolic frame number
                #reading the end-systolic 3d images:
                if int(s)<10:
                    im_ESframe= sitk.ReadImage('{}\{}\{}_frame0{}.nii.gz'.format(path,name,name,s))
                    im_ESgt=sitk.ReadImage('{}\{}\{}_frame0{}.nii.gz'.format(path,name,name,s))
                else:
                    im_ESframe= sitk.ReadImage('{}\{}\{}_frame{}.nii.gz'.format(path,name,name,s))
                    im_ESgt=sitk.ReadImage('{}\{}\{}_frame{}_gt.nii.gz'.format(path,name,name,s))
                
        #Converting the 3d images into 3 dimensional arrays:        
        arr_EDframe= sitk.GetArrayFromImage(im_EDframe)
        arr_EDgt= sitk.GetArrayFromImage(im_EDgt)
        arr_ESframe= sitk.GetArrayFromImage(im_ESframe)
        arr_ESgt= sitk.GetArrayFromImage(im_ESgt)
    
        NSlices=arr_EDframe.shape[0]
     
        #l=list with the patient number, the number of slices per frame, and the four 3D frames as arrays
        l.append([i+1, NSlices,arr_EDframe,arr_EDgt,arr_ESframe,arr_ESgt])
        
    return l


# In[ ]:


#This function gives as output the slices of end-diastolic and end-systolic frames of 100 patients, including both the initial 
#and segmented images. If the output is the list Y:
#Y[0] = patient number
#Y[1] = slice number
#Y[2] = ED 2D original image AS AN ARRAY
#Y[3] = ED 2D ground truth AS AN ARRAY
#Y[4] = ES 2D original image AS AN ARRAY
#Y[5] = ES 2D ground truth AS AN ARRAY

def load2Ddata(path):
    #"path" is the path of the folder named "Data"
    l=load3Ddata(path)
    output=[]
    for i in range(100):
        n=l[i][1]
        for h in range(n):
            output.append([l[i][0],h+1,l[i][2][h],l[i][3][h],l[i][4][h],l[i][5][h]])
    return output
            


# In[ ]:


#This code gives as output the slices of end-diastolic and end-systolic frames of 100 patients, including both the initial 
#and segmented images, PADDED SO ALL ARRAYS HAVE THE SAME SIZE. If the output is the list Y:
#Y[0] = patient number
#Y[1] = slice number
#Y[2] = ED 2D original image AS AN ARRAY
#Y[3] = ED 2D ground truth AS AN ARRAY
#Y[4] = ES 2D original image AS AN ARRAY
#Y[5] = ES 2D ground truth AS AN ARRAY

def load2Ddata_resized(path):
    #"path" is the path of the folder named "Data"
    output=load2Ddata(path)
    for i in range(951):
        if output[i][2].shape[0]<512:    #the bigger images have size 512x428
            L0=512-output[i][2].shape[0]
            L1=428-output[i][2].shape[1]
            output[i][2]=np.pad(output[i][2], ((0,L0),(0,L1)), 'constant', constant_values=0)
            output[i][3]=np.pad(output[i][3], ((0,L0),(0,L1)), 'constant', constant_values=0)
            output[i][4]=np.pad(output[i][4], ((0,L0),(0,L1)), 'constant', constant_values=0)
            output[i][5]=np.pad(output[i][5], ((0,L0),(0,L1)), 'constant', constant_values=0)
    return output
            

