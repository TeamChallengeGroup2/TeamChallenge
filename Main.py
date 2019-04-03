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
from Cropping import cropROI, cropImage
from fcnNetwork import fcn_model
from Validate import plotResults, metrics
from DataAugmentation import create_Augmented_Data

# -----------------------------------------------------------------------------
# INPUT
path = r'C:\Users\s154150\Desktop\Team Challenge\TeamChallenge'
networkpath = r'C:\Users\s154150\Desktop\Team Challenge\TeamChallenge\augmentation_trainednetwork.h5'
nr_of_batches_augmentation = 30
batchsize_augmentation = 25
batchsize = 5
epochs = 20
trainnetwork = False
testing = True
plot = True
augmentation = True
EjectionFraction = False

# -----------------------------------------------------------------------------
# LOADING THE DATA
data_original=loadData(path)
print('Data Loaded')

# Shuffle the data to take a random subset for training later
random.shuffle(data_original)

# Split up the data set into a training and test set. Note that the validation frames
# are included in the Train_frames array
Train_frames = data_original[:3*int(len(data_original)/4)]
Test_frames = data_original[int(len(data_original)-len(data_original)/4):]

# Shuffle the training data to take a random subset for training later
random.shuffle(Train_frames)

Train_frames_cropped = cropImage(Train_frames)
Test_frames_cropped = cropImage(Test_frames)

# Extract the frames and spacing
framesTrain = []
groundtruthTrain = []
spacingsTrain = []

framesTest = []
groundtruthTest = []
spacingsTest = []

for i in range(len(Train_frames_cropped)):    
    # Append the ED frame 
    framesTrain.append(Train_frames_cropped[i][2])
    # Append the ES frame
    framesTrain.append(Train_frames_cropped[i][4])
    # Append the ED groundtruth 
    groundtruthTrain.append(Train_frames_cropped[i][3])
    # Append the ES groundtruth
    groundtruthTrain.append(Train_frames_cropped[i][5])
    # Append the spacing
    spacingsTrain.append(Train_frames_cropped[i][6])
    
for i in range(len(Test_frames_cropped)):    
    # Append the ED frame 
    framesTest.append(Test_frames_cropped[i][2])
    # Append the ES frame
    framesTest.append(Test_frames_cropped[i][4])
    # Append the ED groundtruth 
    groundtruthTest.append(Test_frames_cropped[i][3])
    # Append the ES groundtruth
    groundtruthTest.append(Test_frames_cropped[i][5])
    # Append the spacing
    spacingsTest.append(Test_frames_cropped[i][6])

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

# -----------------------------------------------------------------------------
# TRAINING
if trainnetwork:
    
    # Train the network
    print ('Start training')
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
    cnn.load_weights('augmentation_trainednetwork.h5')
    
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