#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import os
import cv2
import SimpleITK as sitk
from keras import backend as K
from scipy.ndimage import label as ndlabel
K.set_image_data_format('channels_last')  # TF dimension ordering in this code


# In[6]:


def EjectionFraction(maskED,maskES,voxelvolume):
    # Compute the stroke volume from the end-diastolic (ED) and end-systolic (ES) volume
    maxn=maskED.max()
    maskED=np.where(maskED>=(maxn-0.2),1,0)
    maskES=np.where(maskES>=(maxn-0.2),1,0)
    ED_volume = (maskED.sum())*voxelvolume
    ES_volume = (maskES.sum())*voxelvolume
    strokevolume = ED_volume - ES_volume
    
    # Compute the Ejection fraction
    LV_EF = (strokevolume/ED_volume)*100
    
    return LV_EF

def biggest_region_3D(array):
    if len(array.shape)==4:
        im_np=np.squeeze(array)
    else:
        im_np=array
    struct=np.full((3,3,3),1)
    c=0
    arr=np.where(im_np>=maxi,1,0)
    lab, num_reg=ndlabel(arr,structure=struct)
    h=np.zeros(num_reg+1)
    for i in range(num_reg):
        z=np.where(lab==(i+1),1,0)
        h[i+1]=z.sum()
        if h[i+1]==h.max():
            c=i+1
    lab=np.where(lab==c,1,0)
    return lab

def respace(itk_image, new_spacing):
    spacing = itk_image.GetSpacing()
    size = itk_image.GetSize()
    new_size = (np.round(size*(spacing/np.array(new_spacing)))).astype(int).tolist()
    new_image = sitk.Resample(itk_image, new_size, sitk.Transform(),sitk.sitkNearestNeighbor,
                            itk_image.GetOrigin(), new_spacing, itk_image.GetDirection(), 0.0,
                            itk_image.GetPixelID())
    return new_image


# In[ ]:


def cropROI(arrayin, slicenr):
    #input: an array with indices [slicenr, imageX, imageY]
    # slicenr: averaging over this number of slices
    
    LVradius=20 #the radius used in frst, which should be the radius of the LV in the top image
    cropdiam=64 #the length in X and Y direction of the cropped image
    multi_mindist = []
    multi_circles = []
    sumdist1 = 0
    sumdist2 = 0
    
    for i in range(slicenr): 
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


# In[5]:


#you have to import first the fcn_model!!!
from fcnNetwork import fcn_model

def show_results(datapath, weightspath):
    
    print('*'*20)
    print('Preparing the model')
    print('*'*20)
    cnn = fcn_model((128,128,1),2,weights=None)
    cnn.load_weights(weightspath)
    
    print('*'*20)
    print('Loading data and predicting')
    print('*'*20)
    path=datapath
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
        spacing=im_EDframe.GetSpacing()
        voxelvolume=spacing[0]*spacing[1]*spacing[2]
        
        LV_EFgt=EjectionFraction(sitk.GetArrayFromImage(im_EDframe),sitk.GetArrayFromImage(im_ESframe),voxelvolume)
        
        z=spacing[2]
        new_spacing=(1.0,1.0,z)
    
    
        #Apply the same spacing in the X and Y dimensions to the images
        im_EDframe=respace(im_EDframe,new_spacing)
        im_EDgt=respace(im_EDgt,new_spacing)
        im_ESframe=respace(im_ESframe,new_spacing)
        im_ESgt=respace(im_ESgt,new_spacing)
        
        
        
        #Converting the 3d images into 3 dimensional arrays:        
        arr_EDframe= sitk.GetArrayFromImage(im_EDframe)
        arr_EDgt= sitk.GetArrayFromImage(im_EDgt)
        arr_ESframe= sitk.GetArrayFromImage(im_ESframe)
        arr_ESgt= sitk.GetArrayFromImage(im_ESgt)
        
        arr_EDgt=np.where(arr_EDgt==3,1,0)
        arr_ESgt=np.where(arr_ESgt==3,1,0)
        
        #Crop the images
        #Tune av_n to improve cropping
        av_n=int((arr_EDframe.shape[0])*3/4)
        
        EDx1, EDx2, EDy1, EDy2=cropROI(arr_EDframe,av_n)

        cropEDgt=arr_EDgt[:,EDx1:EDx2, EDy1:EDy2]
        cropESgt=arr_ESgt[:,EDx1:EDx2, EDy1:EDy2]
        cropEDframe=arr_EDframe[:,EDx1:EDx2, EDy1:EDy2]
        cropESframe=arr_ESframe[:,EDx1:EDx2, EDy1:EDy2]
        
        array=cropEDframe[:,:,:,np.newaxis]
        maskED = cnn.predict(array, verbose=1)
        maskED =np.squeeze(maskED)
        maskEDB=biggest_region_3D(maskED)
        
        array=cropESframe[:,:,:,np.newaxis]
        maskES = cnn.predict(array, verbose=1)
        maskES =np.squeeze(maskES)
        maskESB=biggest_region_3D(maskES)
        
        for j in range(arr_EDgt.shape[0]):
            if arr_EDgt[j].sum()==0:
                maskED[j]=arr_EDgt[j]
                maskEDB[j]=arr_EDgt[j]
            if arr_ESgt[j].sum()==0:
                maskES[j]=arr_ESgt[j]
                maskESB[j]=arr_ESgt[j]
    
        # Compute the Ejection fraction of the segmentations
        LV_EF = EjectionFraction(maskED,maskES,z)
        LV_EFB = EjectionFraction(maskEDB,maskESB,z)
        
        print('Patient ',i+1,'\nGround truth: ',LV_EFgt,', prediction: ',LV_EF,', postprocessed: ',LV_EFB)
    


# In[ ]:


show_results(r'Data','fcn_weights.h5')

