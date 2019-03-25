# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:39:37 2019
Team Challenge (TU/e & UU)
Team 2
"""

import numpy as np
import keras
import time
import random
import cv2
import matplotlib.pyplot as plt
import itertools
import scipy
import os

from Data import loadData
from Augmentation import augmentation
from Cropping import cropROI
from Network import buildUnet
from fcnNetwork import fcn_model
from Patches import make2Dpatches, make2Dpatchestest
from Validate import plotResults, calculateDice, metrics
from DICEscore import DSC


# -----------------------------------------------------------------------------
# INPUT
path = os.path.realpath(__file__).replace("\\Main.py","")
networkpath = r'trainednetwork.h5'
nr_augmentations = 30
minibatches = 1000
minibatchsize = 100
patchsize = 32
trainnetwork = False
validation = True
plot = False

# -----------------------------------------------------------------------------
# LOADING THE DATA
data_original=loadData(path)
print('Data Loaded')

# -----------------------------------------------------------------------------
# DATA AUGMENTATION
data=augmentation(data_original,nr_augmentations)
print('Augmentation succeeded')

# -----------------------------------------------------------------------------
# CROPPING
output=[] # List with the patient number, the slices number, and the four 2D slices as arrays
slice_count=[0]

for j in range(len(data)):

    # Extract the ED frame and ES frame for each patient, separately
    EDframe=data[j][2]
    ESframe=data[j][4]
    
    # Crop only if HoughCircles is able to find a circle
    cropped_EDim, EDx1, EDx2, EDy1, EDy2=cropROI(EDframe,4)
    cropped_ESim, ESx1, ESx2, ESy1, ESy2=cropROI(ESframe,4)
    if cropped_EDim.size and cropped_ESim.size:
        # Extract the slice number
        n=data[j][1]
        # Extract and save the ED and ES slices and ground truth slices
        for h in range(n):
            EDslice=data[j][2][h]
            EDslicegt=data[j][3][h]
            ESslice=data[j][4][h]
            ESslicegt=data[j][5][h]
            
            # Save the data in lists
            output.append([data[j][0],h+1,EDslice[EDx1:EDx2, EDy1:EDy2],EDslicegt[EDx1:EDx2, EDy1:EDy2],ESslice[ESx1:ESx2, ESy1:ESy2],ESslicegt[ESx1:ESx2, ESy1:ESy2],data[j][6]])
            slice_count.append(slice_count[j]+n)
            
print('Images cropped')

# -----------------------------------------------------------------------------
# TRAINING

# Shuffle the data to take a random subset for training later
random.shuffle(output)

# Split the list l into a list containing all ED frames
frames = []
groundtruth = []
spacings = []

for i in range(len(output)):    
    # Append the ED frame 
    frames.append(output[i][2])
    # Append the ES frame
    frames.append(output[i][4])
    # Append the ED groundtruth 
    groundtruth.append(output[i][3])
    # Append the ES groundtruth
    groundtruth.append(output[i][5])
    # Append the spacing
    spacings.append(output[i][6])

# Convert the lists containing the frames and groundtruth to arrays
frames=np.array(frames)
groundtruth=np.array(groundtruth)

# Pad the images with zeros to allow patch extraction at all locations
halfsize = int(patchsize/2)    
frames = np.pad(frames,((0,0),(halfsize,halfsize),(halfsize,halfsize)),'constant', constant_values=0)
groundtruth = np.pad(groundtruth,((0,0),(halfsize,halfsize),(halfsize,halfsize)),'constant', constant_values=0)
    
# Split up the data set into a training, test and validation set
Train_frames = frames[:int(len(frames)/2)]
Valid_frames = frames[int(len(frames)/2):int(len(frames)-len(frames)/4)]
Test_frames = frames[int(len(frames)-len(frames)/4):]

Train_frames = np.array(Train_frames)
Valid_frames = np.array(Valid_frames)
Test_frames = np.array(Test_frames)

Train_labels = groundtruth[:int(len(groundtruth)/2)]
Valid_labels = groundtruth[int(len(groundtruth)/2):int(len(groundtruth)-len(groundtruth)/4)]
Test_labels = groundtruth[int(len(groundtruth)-len(groundtruth)/4):]

Train_labels = np.array(Train_labels)
Valid_labels = np.array(Valid_labels)
Test_labels = np.array(Test_labels)

# Initialise the network
#cnn = buildUnet()
cnn = fcn_model((32,32,1),2,weights='fcn_weights.h5')

# Seperately select the positive samples (Left ventricle) and negative samples (background)
positivesamples = np.nonzero(Train_labels)
negativesamples = np.nonzero(Train_frames-Train_labels)

validsamples=np.where(Valid_labels==3)

# Train the network
if trainnetwork:
    trainlosslist = []
    validlosslist = []
    probabilities = np.empty((0,))
    t0 = time.time()

    for i in range(minibatches):
        # Take random samples
        posbatch = random.sample(list(range(len(positivesamples[0]))),int(minibatchsize/2))
        negbatch = random.sample(list(range(len(negativesamples[0]))),int(minibatchsize/2))
        
        valid_batch = random.sample(list(range(len(validsamples[0]))), int(minibatchsize/2))
        
        # Make the patches
        Xpos, Ypos = make2Dpatches(positivesamples,posbatch,Train_frames,patchsize,1) # double patchsize for rotation
        Xneg, Yneg = make2Dpatches(negativesamples,negbatch,Train_frames,patchsize,0)   # it is cropped later
        
        x_valid, y_valid = make2Dpatches(validsamples, valid_batch, Valid_frames, patchsize, 1)
        
        # Concatenate the positive and negative patches
        Xtrain = np.vstack((Xpos,Xneg))
        Ytrain = np.vstack((Ypos,Yneg))

        # Perform the training and compute the training loss
        train_loss = cnn.train_on_batch(Xtrain,Ytrain)
        trainlosslist.append(train_loss[0])
        
        # Perform the validation and compute the validation loss
        valid_loss = cnn.test_on_batch(x_valid, y_valid)
        validlosslist.append(valid_loss[0])
        
        print('Batch: {}'.format(i))
        print('Train Loss: {} \t Train Accuracy: {}'.format(train_loss[0], train_loss[1]))
        print('Valid loss: {} \t Valid Accuracy: {}'.format(valid_loss[0], valid_loss[1]))
        
    # Save the network
    cnn.save(networkpath)
    t1 = time.time()
    print('\nTraining time: {} seconds'.format(t1 - t0))
    
    plt.figure()
    plt.plot(trainlosslist)
    plt.plot(validlosslist)
    
else:
    # Load the network
    cnn = keras.models.load_model(networkpath)
    
    
print ('Training is finished')

# -----------------------------------------------------------------------------
# VALIDATION

if validation:
    
    # Number of patients to validate
    valsamples=4
    
    # All indices for sample_idx are the ED frames and sample_idx+1 are the ES frames
    idx=np.multiply(2,random.sample(range(len(Valid_frames)//2), valsamples))
    sample_idx=list(itertools.chain(*zip(idx, idx++1)))
    
    Valid_frames=Valid_frames[sample_idx]
    Valid_labels=Valid_labels[sample_idx]
    
    probimage = np.zeros(Valid_frames.shape)
    
    # Loop through all frames in the validation set
    for j in range(np.shape(Valid_frames)[0]):
        print('Image {} of {}'. format(j+1, np.shape(Valid_frames)[0]))
        
        # Take all labels of the Left ventricle (3) or all structures together
        validsamples = np.where(Valid_labels[j]==3)
        probabilities = np.empty((0,))
        
        # Define the minibatchsize, it can be as large as the memory allows
        minibatchsize = 100
    
        # Loop through all samples
        for k in range(0,len(validsamples[0]),minibatchsize):
            print('{}/{} samples labelled'.format(k,len(validsamples[0])))
        
            # Determine the batches for the validation
            if k+minibatchsize < len(validsamples[0]):
                valbatch = np.arange(k,k+minibatchsize)        
            else:
                valbatch = np.arange(k,len(validsamples[0]))        
                    
            # Make the patches
            Xval = make2Dpatchestest(validsamples,valbatch,Valid_frames[j],patchsize)

            # Compute the probability
            prob = cnn.predict(Xval, batch_size=minibatchsize)
            probabilities = np.concatenate((probabilities,prob[:,1]))
    
        # Create the probability image        
        for m in range(len(validsamples[0])):
            probimage[j,validsamples[0][m],validsamples[1][m]] = probabilities[m]
            
            # Convert the probability to a binary mask with threshold 0.5
            threshold,mask = cv2.threshold(probimage,0.5,1.0,cv2.THRESH_BINARY)
           
            mask=scipy.ndimage.binary_closing(mask).astype(np.int)
    
    # Compute the DICE coefficient, accuracy, sensitivity and specificity per image
    dices = DSC(mask, Valid_labels)
    Accuracy, Sensitivity, Specificity = metrics(mask, Valid_labels)
    
    # Plot the results
    if plot:
        plotResults(Valid_frames, Valid_labels, mask)
    
# -----------------------------------------------------------------------------
# EJECTION FRACTION
EF=[]

for k in range(0,len(probimage),2):
    # Determine the voxelvolume
    voxelvolume = spacings[k][0]*spacings[k][1]*spacings[k][2]
    
    # Compute the stroke volume from the end-diastolic (ED) and end-systolic (ES) volume
    ED_volume = np.sum(mask[k,:,:]==1)*voxelvolume
    ES_volume = np.sum(mask[k+1,:,:]==1)*voxelvolume
    strokevolume = ED_volume - ES_volume
    
    # Compute the Ejection fraction per patient and save in a list
    LV_EF = (strokevolume/ED_volume)*100
    EF.append(LV_EF)