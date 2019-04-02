"""
Main script

Team Challenge (TU/e & UU)
Team 2
"""

import numpy as np
import keras
import time
import random
import os
import cv2
import matplotlib.pyplot as plt

from Data import loadData
from Cropping import cropROI
from fcnNetwork import fcn_model
from Validate import plotResults, metrics
from DataAugmentation import create_Augmented_Data

# -----------------------------------------------------------------------------
# INPUT
path = r'C:\Users\s141352\Documents\BMT\Master\Team Challenge\Part 2'
networkpath = r'trainednetwork.h5'
nr_of_batches_augmentation = 30
batchsize_augmentation = 25
batchsize = 5
epochs = 10
trainnetwork = True
testing = True
plot = False
augmentation = True
EjectionFraction = False

# -----------------------------------------------------------------------------
# LOADING THE DATA
data_original=loadData(path)
print('Data Loaded')

# -----------------------------------------------------------------------------
# CROPPING
data=[] # List with the patient number, the slices number, and the four 2D slices as arrays
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
            data.append([data_original[j][0],h+1,EDslice[EDx1:EDx2, EDy1:EDy2],EDslicegt[EDx1:EDx2, EDy1:EDy2],ESslice[ESx1:ESx2, ESy1:ESy2],ESslicegt[ESx1:ESx2, ESy1:ESy2],data_original[j][6]])
            
            
print('Images cropped')

# -----------------------------------------------------------------------------
# TRAINING

# Shuffle the data to take a random subset for training later
random.shuffle(data)

# Split up the data set into a training and test set. Note that the validation frames
# are included in the Train_frames array
Train_frames = data[:3*int(len(data)/4)]
Test_frames = data[int(len(data)-len(data)/4):]


# Split the list l into a list containing all ED frames
framesTrain = []
groundtruthTrain = []
spacingsTrain = []

framesTest = []
groundtruthTest = []
spacingsTest = []

for i in range(len(Train_frames)):    
    # Append the ED frame 
    framesTrain.append(Train_frames[i][2])
    # Append the ES frame
    framesTrain.append(Train_frames[i][4])
    # Append the ED groundtruth 
    groundtruthTrain.append(Train_frames[i][3])
    # Append the ES groundtruth
    groundtruthTrain.append(Train_frames[i][5])
    # Append the spacing
    spacingsTrain.append(Train_frames[i][6])
    
for i in range(len(Test_frames)):    
    # Append the ED frame 
    framesTest.append(Test_frames[i][2])
    # Append the ES frame
    framesTest.append(Test_frames[i][4])
    # Append the ED groundtruth 
    groundtruthTest.append(Test_frames[i][3])
    # Append the ES groundtruth
    groundtruthTest.append(Test_frames[i][5])
    # Append the spacing
    spacingsTest.append(Test_frames[i][6])

framesTrain = np.array(framesTrain)
framesTest = np.array(framesTest)

groundtruthTrain = np.array(groundtruthTrain)
groundtruthTest = np.array(groundtruthTest)

# Scale the masks to binary masks
groundtruthTrain = cv2.threshold(groundtruthTrain,2.5,1.0,cv2.THRESH_BINARY)[1]
groundtruthTest = cv2.threshold(groundtruthTest,2.5,1.0,cv2.THRESH_BINARY)[1]

# Augment the data
if augmentation:
    framesTrain, groundtruthTrain = create_Augmented_Data(framesTrain, groundtruthTrain, batchsize_augmentation, nr_of_batches_augmentation)

# Initialize the model
cnn  = fcn_model((126,126,1),2,weights=None)

# Train the network
print ('Start training')

if trainnetwork:
    t0 = time.time()
    
    # Train the network 
    hist=cnn.fit(framesTrain[:,:,:,np.newaxis], groundtruthTrain[:,:,:,np.newaxis],
            batch_size=batchsize, epochs=epochs, verbose=1, shuffle=True, validation_split = 0.25)
    
    # Save the network
    cnn.save(networkpath)
    t1 = time.time()
    print('\nTraining time: {} seconds'.format(t1 - t0))
    
else:
    # Load the network
    cnn = keras.models.load_model(networkpath)
    
print ('Training is finished')

# -----------------------------------------------------------------------------
# VALIDATION
print ('Start testing')

if testing:
    
    # Predict the masks
    mask = cnn.predict(framesTest[:,:,:,np.newaxis], verbose=1)

    # As the prediction have the channels dimension (3th dimension per slice), 
    # to go back to 2 dimensions per slice:
    mask=np.squeeze(mask)
    
    # Compute the DICE coefficient, accuracy, sensitivity and specificity per image
    Dice, Accuracy, Sensitivity, Specificity = metrics(mask, groundtruthTest)
    
    # Plot the results
    if plot:
        plotResults(framesTest, groundtruthTest, mask)

print ('Testing is finished')
    
# -----------------------------------------------------------------------------
# EJECTION FRACTION

if EjectionFraction:
    print ('Start computing the Ejection Fraction')
    
    EF=[]
    EF_gt=[]
    
    for k in range(0,len(mask),2):
        # Determine the voxelvolume
        voxelvolume_ED = spacings[k][0]*spacings[k][1]*spacings[k][2]
        voxelvolume_ES = spacings[k+1][0]*spacings[k+1][1]*spacings[k+1][2]
        
        # Compute the stroke volume from the end-diastolic (ED) and end-systolic (ES) volume
        ED_volume = np.sum(mask[k,:,:]==1)*voxelvolume_ED
        ES_volume = np.sum(mask[k+1,:,:]==1)*voxelvolume_ES
        strokevolume = ED_volume - ES_volume
        
        # Compute the Ejection fraction per patient and save in a list
        LV_EF = (strokevolume/ED_volume)*100
        EF.append(LV_EF)
        
        # Ground truth
        ED_volume_gt = np.sum(Test_labels[k,:,:]==3)*voxelvolume_ED
        ES_volume_gt = np.sum(Test_labels[k+1,:,:]==3)*voxelvolume_ES
        strokevolume_gt = ED_volume_gt - ES_volume_gt
        LV_EF_gt = (strokevolume_gt/ED_volume_gt)*100
        EF_gt.append(LV_EF_gt)
        
    print ('Ejection Fraction is computed')