import os
import numpy as np
import SimpleITK as sitk
from frst import frst
import matplotlib.pyplot as plt
import cv2
from PIL import Image

os.chdir('C:/Users/Guus/Desktop/Team challenge/TeamChallenge')

def loadData():
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
    
    return l

#do not get why you create this output list??

def cropROI(arrayin):
    #input: an array with indices [slicenr, imageX, imageY]
    
    LVradius=20 #the radius used in frst, which should be the radius of the LV in the top image
    cropdiam=50 #the length in X and Y direction of the cropped image
    
    
    topslice=arrayin[1,:,:]
    
    
    #im=arrayin[1,:,:]
    #im=Image.fromarray(arrayin[1,:,:], 'L')
    im = np.array((topslice/np.amax(topslice)) * 255, dtype = np.uint8)
    #cv2.imshow('image', im)
    #print(im.shape)
    #plt.figure(1)
    #plt.imshow(im,cmap='gray')
    #plt.show()
            
    #apply fast radil symmetry transform
    #syntax: frst(image, radius, strictness, gradient threshold, gaussian std, mode)
    #returns grayscale image with white = most radially symmetric pixels
    #center=frst.frst(im, LVradius, 0.01,0.5, 1, mode='BRIGHT')
    
    
    
    circles=cv2.HoughCircles(im, cv2.HOUGH_GRADIENT, 1, 100, param1=50, param2=30, minRadius=LVradius-10, maxRadius=LVradius+10)
    
    print(circles)
    
    plt.figure(1)
    plt.imshow(im, cmap='gray')
    
    #print(center.shape)
    #plt.figure(2)
    #plt.imshow(center,cmap='gray')
    #plt.show()
            
    #assume brightest pixel is center of LV
    #ind = np.unravel_index(np.argmax(center, axis=None), center.shape)
    #print(ind)
            
    #crop on centerpoint with cropdiam
    #croppedim=im[ind[0]-cropdiam:ind[0]+cropdiam,ind[1]-cropdiam:ind[1]+cropdiam]
            
    #return croppedim

l=loadData()
patient1=l[0]
EDpatient1=patient1[2]
cropped_im=cropROI(EDpatient1)
