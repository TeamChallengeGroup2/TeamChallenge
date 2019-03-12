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

from Data import loadData
from Cropping import cropROI
from Network import buildUnet
from Patches import make2Dpatches

# -----------------------------------------------------------------------------
# INPUT
networkpath = r'trainednetwork.h5'
minibatches = 50
minibatchsize = 200
patchsize = 32
trainnetwork = True

# -----------------------------------------------------------------------------
# LOADING THE DATA
data=loadData()
print('Data Loaded')

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
            output.append([data[j][0],h+1,EDslice[EDx1:EDx2, EDy1:EDy2],EDslicegt[EDx1:EDx2, EDy1:EDy2],ESslice[ESx1:ESx2, ESy1:ESy2],ESslicegt[ESx1:ESx2, ESy1:ESy2]])
            slice_count.append(slice_count[j]+n)
            
print('Images cropped')

# -----------------------------------------------------------------------------
# TRAINING

# Shuffle the data to take a random subset for training later
random.shuffle(output)

# Split the list l into a list containing all ED frames
EDframes = []
EDground = []
ESframes = []
ESground = []

for i in range(len(output)):
    EDframes.append(output[i][2])
    EDground.append(output[i][3])
    ESframes.append(output[i][4])
    ESground.append(output[i][5])

# Take the ES frames and ED frames 
frames = ESframes+EDframes
groundtruth = ESground+EDground

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
cnn = buildUnet()

# Seperately select the positive samples (Left ventricle) and negative samples (background)
positivesamples = np.nonzero(Train_labels)
negativesamples = np.nonzero(Train_frames-Train_labels)

validsamples=np.where(Valid_labels==3)

# Train the network
if trainnetwork:
    trainlosslist = []
    validlosslist = []
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
        trainlosslist.append(train_loss)
        
        # Perform the validation and compute the validation loss
        valid_loss = cnn.test_on_batch(x_valid, y_valid)
        validlosslist.append(valid_loss)
        
        print('Batch: {}'.format(i))
        print('Train Loss: {} \t Train Accuracy: {}'.format(train_loss[0], train_loss[1]))
        print('Valid loss: {} \t Valid Accuracy: {}'.format(valid_loss[0], valid_loss[1]))
           
    # Save the network
    cnn.save(networkpath)
    t1 = time.time()
    print('\nTraining time: {} seconds'.format(t1 - t0))
    
    plt.figure()
    plt.plot(losslist)
    plt.plot(validlosslist)
    
else:
    # Load the network
    cnn = keras.models.load_model(networkpath)
    
print ('Training is finished')
