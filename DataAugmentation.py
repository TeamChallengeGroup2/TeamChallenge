"""
Data augmentation

Team Challenge (TU/e & UU)
Team 2
"""


import numpy as np
from keras.preprocessing.image import ImageDataGenerator

def create_Augmented_Data(images, segmentations, b_size, numberofbatches):
    
    # Data augmentation parameters
    data_gen_args = dict(rotation_range=90,
                         horizontal_flip = True,
                         vertical_flip = True,
                         )
    
    # Data generators
    image_datagen = ImageDataGenerator(**data_gen_args)
    segmentations_datagen = ImageDataGenerator(**data_gen_args)
    
    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    
    augmented_images = images
    augmented_segmentations = segmentations
    
    # Make the images and segmentations rank 4 so they could be augmentated
    images = np.expand_dims(images, axis=3)
    segmentations = np.expand_dims(segmentations, axis=3)
    
    image_datagen.fit(images, augment=True, seed=seed)
    segmentations_datagen.fit(segmentations, augment=True, seed=seed)

    print('Generate Images')
    batches = 0
    for x_batch in image_datagen.flow(images, batch_size=b_size, seed=seed):
        x_batch = np.squeeze(x_batch)
        augmented_images = np.concatenate((augmented_images,x_batch), axis=0)
        batches += 1
        if batches >= numberofbatches:
            break
        
    print('Generate Segmentations')
    batches = 0
    for y_batch in image_datagen.flow(segmentations, batch_size=b_size, seed=seed):
        y_batch = np.squeeze(y_batch)
        augmented_segmentations = np.concatenate((augmented_segmentations,y_batch), axis=0)
        batches += 1
        if batches >= numberofbatches:
            break
       
    return augmented_images, augmented_segmentations
