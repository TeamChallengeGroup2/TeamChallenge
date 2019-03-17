# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 15:42:19 2019

@author: User
"""

import os
import SimpleITK as sitk
import numpy as np
import keras
import time
import random
import cv2
import math
import matplotlib.pyplot as plt
from numpy import array

def loadData():
    # This function loads the data and save it into a list with the patient 
    # number, the number of slices per frame, and the four 3D frames as arrays
    
    l=[]
    for i, name in enumerate(os.listdir('Data')):
        data = open('Data\{}\Info.cfg'.format(name), 'r')
        
        ED=data.readline()    # End-diastolic frame information
        for s in ED.split():
            if s.isdigit():   # End-diastolic frame number
                # Reading the end-diastolic 3d images:
                if int(s)<10:
                    im_EDframe= sitk.ReadImage('Data\{}\{}_frame0{}.nii.gz'.format(name,name,s))
                    im_EDgt=sitk.ReadImage('Data\{}\{}_frame0{}_gt.nii.gz'.format(name,name,s))
                else:
                    im_EDframe= sitk.ReadImage('Data\{}\{}_frame{}.nii.gz'.format(name,name,s))
                    im_EDgt=sitk.ReadImage('Data\{}\{}_frame{}_gt.nii.gz'.format(name,name,s))
                    
        ES=data.readline()    # End-systolic frame information
        for s in ES.split():
            if s.isdigit():   # End-systolic frame number
                # Reading the end-systolic 3d images:
                if int(s)<10:
                    im_ESframe= sitk.ReadImage('Data\{}\{}_frame0{}.nii.gz'.format(name,name,s))
                    im_ESgt=sitk.ReadImage('Data\{}\{}_frame0{}_gt.nii.gz'.format(name,name,s))
                else:
                    im_ESframe= sitk.ReadImage('Data\{}\{}_frame{}.nii.gz'.format(name,name,s))
                    im_ESgt=sitk.ReadImage('Data\{}\{}_frame{}_gt.nii.gz'.format(name,name,s))
                    
        # Converting the 3d images into 3 dimensional arrays:        
        arr_EDframe= sitk.GetArrayFromImage(im_EDframe)
        arr_EDgt= sitk.GetArrayFromImage(im_EDgt)
        arr_ESframe= sitk.GetArrayFromImage(im_ESframe)
        arr_ESgt= sitk.GetArrayFromImage(im_ESgt)
        
        NSlices=arr_EDframe.shape[0]
        
        # Save all in a list 
        l.append([i+1, NSlices,arr_EDframe,arr_EDgt,arr_ESframe,arr_ESgt])
        
    return l



# -----------------------------------------------------------------------------
    
# LOADING THE DATA
data = loadData()
print('Data Loaded')

# -----------------------------------------------------------------------------

 #DATA AUGMENTATION
 
augmented_data=[]
augmented_slices=[]
sl_count=[0]
 
for i in range(len(data)):
    #extract ED and ES frame for each patient with the ground truths
    ED_frame=data[i][2]
    EDgt_frame=data[i][3]
    ES_frame=data[i][4]
    ESgt_frame=data[i][5]
    #extract slice number
    n=data[i][1]
    #extract slices of ED, ES and ground truths
    for j in range(n):
        ED_slice=data[i][2][j]
        EDgt_slice=data[i][3][j]
        ES_slice=data[i][4][j]
        ESgt_slice=data[i][5][j]
    #flip the slices
        if j<=n//2:
            ED_aug=np.fliplr(ED_slice)
            EDgt_aug=np.fliplr(EDgt_slice)
            ES_aug=np.fliplr(ES_slice)
            ESgt_aug=np.fliplr(ESgt_slice)
            #augmented_slices.append(ED_aug)
    #apply Gaussian blur filter
        if j>n//2 and j<=n:
            ED_aug=cv2.GaussianBlur(ED_slice,(5,5),0) 
            EDgt_aug=cv2.GaussianBlur(EDgt_slice,(5,5),0)
            ES_aug=cv2.GaussianBlur(ES_slice,(5,5),0)
            ESgt_aug=cv2.GaussianBlur(ESgt_slice,(5,5),0)
            #augmented_slices.append(ED_aug)
    #make frames from slices (2D->3D)
    
    #save in list: patient number, slice number, ED, ground truth ED, ES, ground truth ES
    augmented_data.append([data[i][0]+100,j+1,ED_aug,EDgt_aug,ES_aug,ESgt_aug])
    sl_count.append(sl_count[i]+n)

print('Augmentation succeeded')
