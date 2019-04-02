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



def create_Augmented_Data(images, segmentations, b_size, numberofbatches):
#    size_array=2*len(cropped_data)
#    images=np.zeros((size_array,126,126))
#    segmentations=np.zeros((size_array,126,126))
#    j=0
#    for i in range(size_array):
#        if i<len(cropped_data):
#            images[i,:,:]=cropped_data[i][2]
#            segmentations[i,:,:]=cropped_data[i][3]
#        else:
#            images[i,:,:]=cropped_data[j][4]
#            segmentations[i,:,:]=cropped_data[j][5]
#            j+=1
        
 
       
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
    for x_batch in image_datagen.flow(images, batch_size=b_size, seed=seed):
        print('Batch: ', batches)
        x_batch=np.squeeze(x_batch)
        augmented_images=np.concatenate((augmented_images,x_batch), axis=0)
        batches+=1
        if batches >= numberofbatches:
            break
        
       
    print('Generate Segmentations')
    batches=0
    for y_batch in image_datagen.flow(segmentations, batch_size=b_size, seed=seed):
        print('Batch: ', batches)
        y_batch=np.squeeze(y_batch)
        augmented_segmentations=np.concatenate((augmented_segmentations,y_batch), axis=0)
        batches+=1
        if batches >= numberofbatches:
            break
       
    return augmented_images, augmented_segmentations

#
#plt.figure()
#plt.imshow(augmented_images[1802,:,:], cmap='gray')
#plt.show()
#
#plt.figure()
#plt.imshow(augmented_segmentations[1802,:,:], cmap='gray')
#plt.show()