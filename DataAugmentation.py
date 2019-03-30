# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 11:43:44 2019

@author: s141352
"""

import numpy as np
import PIL.Image
import glob
import copy
from keras.preprocessing.image import ImageDataGenerator
import os
from shutil import copyfile
from Data import loadData
from Cropping import cropROI
import matplotlib.pyplot as plt



def create_Augmented_Data(data_original):
    # CROPPING
    cropped_data=[] # List with the patient number, the slices number, and the four 2D slices as arrays
    slice_count=[]
    
    for j in range(len(data_original)):
    
        # Extract the ED frame and ES frame for each patient, separately
        EDframe=data_original[j][2]
        ESframe=data_original[j][4]
        
        # Crop only if HoughCircles is able to find a circle
        cropped_EDim, EDx1, EDx2, EDy1, EDy2=cropROI(EDframe,4)
        cropped_ESim, ESx1, ESx2, ESy1, ESy2=cropROI(ESframe,4)
        if cropped_EDim.size and cropped_ESim.size:
            # Extract the slice number
            n=data_original[j][1]
            slice_count.append(n)
            # Extract and save the ED and ES slices and ground truth slices
            for h in range(n):
                EDslice=data_original[j][2][h]
                EDslicegt=data_original[j][3][h]
                ESslice=data_original[j][4][h]
                ESslicegt=data_original[j][5][h]
                
                # Save the data in lists
                cropped_data.append([data_original[j][0],h+1,EDslice[EDx1:EDx2, EDy1:EDy2],EDslicegt[EDx1:EDx2, EDy1:EDy2],ESslice[ESx1:ESx2, ESy1:ESy2],ESslicegt[ESx1:ESx2, ESy1:ESy2],data_original[j][6]])
                
                
    print('Images cropped')
    
    
    size_array=2*len(cropped_data)
    images=np.zeros((size_array,126,126))
    segmentations=np.zeros((size_array,126,126))
    j=0
    for i in range(size_array):
        if i<len(cropped_data):
            images[i,:,:]=cropped_data[i][2]
            segmentations[i,:,:]=cropped_data[i][3]
        else:
            images[i,:,:]=cropped_data[j][4]
            segmentations[i,:,:]=cropped_data[j][5]
            j+=1
        
 
       
    # Data augmentation parameters
    data_gen_args = dict(featurewise_center=True,
                         featurewise_std_normalization=True,
                         rotation_range=90,
                         horizontal_flip = True,
                         vertical_flip = True,
                         brightness_range=[0.9,1.1]
                         )
    
    # Data generators
    image_datagen = ImageDataGenerator(**data_gen_args)
    segmentations_datagen = ImageDataGenerator(**data_gen_args)
    
    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    
    augmented_images=images
    augmented_segmentations=segmentations
    # make the images and segmentations rank 4 so they could be augmentated
    images=np.expand_dims(images, axis=3)
    segmentations=np.expand_dims(segmentations, axis=3)
    
    image_datagen.fit(images, augment=True, seed=seed)
    segmentations_datagen.fit(segmentations, augment=True, seed=seed)

    print('Generate Images')
    batches=0
    for x_batch in image_datagen.flow(images, batch_size=25, seed=seed):
        print('Batch: ', batches)
        x_batch=np.squeeze(x_batch)
        augmented_images=np.concatenate((augmented_images,x_batch), axis=0)
        batches+=1
        if batches >=3:
            break
        
       
    print('Generate Segmentations')
    batches=0
    for y_batch in image_datagen.flow(segmentations, batch_size=25, seed=seed):
        print('Batch: ', batches)
        y_batch=np.squeeze(y_batch)
        augmented_segmentations=np.concatenate((augmented_segmentations,y_batch), axis=0)
        batches+=1
        if batches >=3:
            break
       
    return augmented_images, augmented_segmentations

path=r'C:\Users\s141352\Documents\BMT\Master\Team Challenge\Part 2'
data_original=loadData(path)
augmented_images, augmented_segmentations=create_Augmented_Data(data_original)


plt.figure()
plt.imshow(augmented_images[1802,:,:], cmap='gray')
plt.show()

plt.figure()
plt.imshow(augmented_segmentations[1802,:,:], cmap='gray')
plt.show()