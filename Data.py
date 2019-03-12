# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:43:05 2019

Team Challenge (TU/e & UU)
Team 2
"""

import os
import SimpleITK as sitk

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
                    im_ESgt=sitk.ReadImage('Data\{}\{}_frame0{}.nii.gz'.format(name,name,s))
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
