"""
Loading the data and image processing

Team Challenge (TU/e & UU)
Team 2
"""

import numpy as np
import os
import SimpleITK as sitk
from scipy import ndimage
from scipy.ndimage import label as ndlabel


def respace(itk_image, new_spacing):
    # This function respaces the images
    spacing = itk_image.GetSpacing()
    size = itk_image.GetSize()
    new_size = (np.round(size*(spacing/np.array(new_spacing)))).astype(int).tolist()
    new_image = sitk.Resample(itk_image, new_size, sitk.Transform(),sitk.sitkNearestNeighbor,
                            itk_image.GetOrigin(), new_spacing, itk_image.GetDirection(), 0.0,
                            itk_image.GetPixelID())
    return new_image

def biggest_region_3D(array):
    if len(array.shape)==4:
        im_np=np.squeeze(array)
    else:
        im_np=array
    struct=np.full((3,3,3),1)
    c=0
    maxn=im_np.max()
    arr=np.where(im_np>=(maxn-0.2),1,0)
    lab, num_reg=ndlabel(arr,structure=struct)
    h=np.zeros(num_reg+1)
    for i in range(num_reg):
        z=np.where(lab==(i+1),1,0)
        h[i+1]=z.sum()
        if h[i+1]==h.max():
            c=i+1
    lab=np.where(lab==c,1,0)
    return lab

def loadData(datapath):
    # This function loads the data and save it into a list with the patient 
    # number, the number of slices per frame, and the four 3D frames as arrays
    os.chdir(datapath)
    l = []
    spacings = []
    for i, name in enumerate(os.listdir('Data')):
        data = open('Data\{}\Info.cfg'.format(name), 'r')
        
        ED = data.readline()    # End-diastolic frame information
        for s in ED.split():
            if s.isdigit():   # End-diastolic frame number
                # Reading the end-diastolic 3d images:
                if int(s)<10:
                    im_EDframe = sitk.ReadImage('Data\{}\{}_frame0{}.nii.gz'.format(name,name,s))
                    im_EDgt = sitk.ReadImage('Data\{}\{}_frame0{}_gt.nii.gz'.format(name,name,s))
                else:
                    im_EDframe = sitk.ReadImage('Data\{}\{}_frame{}.nii.gz'.format(name,name,s))
                    im_EDgt = sitk.ReadImage('Data\{}\{}_frame{}_gt.nii.gz'.format(name,name,s))
                    
        ES = data.readline()    # End-systolic frame information
        for s in ES.split():
            if s.isdigit():   # End-systolic frame number
                # Reading the end-systolic 3d images:
                if int(s)<10:
                    im_ESframe = sitk.ReadImage('Data\{}\{}_frame0{}.nii.gz'.format(name,name,s))
                    im_ESgt = sitk.ReadImage('Data\{}\{}_frame0{}_gt.nii.gz'.format(name,name,s))
                else:
                    im_ESframe = sitk.ReadImage('Data\{}\{}_frame{}.nii.gz'.format(name,name,s))
                    im_ESgt = sitk.ReadImage('Data\{}\{}_frame{}_gt.nii.gz'.format(name,name,s))
        
        # The spacings of the ES and ED frames are equal
        spacing = im_ESframe.GetSpacing() 
        spacings.append(spacing)
        
        z = spacing[2]
        new_spacing = (1.0,1.0,z)
        
        if im_EDframe.GetSpacing() != im_EDgt.GetSpacing() or im_ESframe.GetSpacing() != im_ESgt.GetSpacing():
            im_EDgt.SetSpacing(im_EDframe.GetSpacing())
            im_ESgt.SetSpacing(im_ESframe.GetSpacing())
        
        im_EDframe = respace(im_EDframe,new_spacing)
        im_EDgt = respace(im_EDgt,new_spacing)
        im_ESframe = respace(im_ESframe,new_spacing)
        im_ESgt = respace(im_ESgt,new_spacing)
    
        
        # Converting the 3d images into 3 dimensional arrays:        
        arr_EDframe = sitk.GetArrayFromImage(im_EDframe)
        arr_EDgt = sitk.GetArrayFromImage(im_EDgt)
        arr_ESframe = sitk.GetArrayFromImage(im_ESframe)
        arr_ESgt = sitk.GetArrayFromImage(im_ESgt)
        
        NSlices = arr_EDframe.shape[0]
        
        # Save all in a list 
        l.append([i+1, NSlices,arr_EDframe,arr_EDgt,arr_ESframe,arr_ESgt,spacing])
        
    return l
