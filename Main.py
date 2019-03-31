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

from Data import loadData
from Augmentation import augmentation
from Cropping import cropROI
from fcnNetwork import fcn_model
from Validate import plotResults, metrics

# -----------------------------------------------------------------------------
# INPUT
path = os.path.realpath("Main.py").replace("\\Main.py","")
networkpath = r'trainednetwork.h5'
nr_augmentations = 30
minibatches = 1000
minibatchsize = 100
patchsize = 32
trainnetwork = True
testing = True
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
    
# Split up the data set into a training and test set. Note that the validation frames
# are included in the Train_frames array
Train_frames = frames[:3*int(len(frames)/4)]
Test_frames = frames[int(len(frames)-len(frames)/4):]

Train_frames = np.array(Train_frames)
Test_frames = np.array(Test_frames)

Train_labels = groundtruth[:3*int(len(groundtruth)/4)]
Test_labels = groundtruth[int(len(groundtruth)-len(groundtruth)/4):]

Train_labels = np.array(Train_labels)
Test_labels = np.array(Test_labels)

# Scale the masks to binary masks
Train_labels = cv2.threshold(Train_labels,2.5,1.0,cv2.THRESH_BINARY)[1]
Test_labels = cv2.threshold(Test_labels,2.5,1.0,cv2.THRESH_BINARY)[1]

# Initialize the model
cnn  = fcn_model((158,158,1),2,weights=None)

# Train the network
print ('Start training')

if trainnetwork:
    t0 = time.time()
    
    # Train the network 
    hist=cnn.fit(Train_frames[:,:,:,np.newaxis], Train_labels[:,:,:,np.newaxis],
            batch_size=5, epochs=1, verbose=1, shuffle=True, validation_split = 0.25)
    
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
    mask = cnn.predict(Test_frames[:,:,:,np.newaxis], verbose=1)

    # As the prediction have the channels dimension (3th dimension per slice), 
    # to go back to 2 dimensions per slice:
    mask=np.squeeze(mask)
    
    # Compute the DICE coefficient, accuracy, sensitivity and specificity per image
    Dice, Accuracy, Sensitivity, Specificity = metrics(mask, Test_labels)
    
    # Plot the results
    if plot:
        plotResults(Test_frames, Test_labels, mask)

print ('Testing is finished')
    
# -----------------------------------------------------------------------------
# EJECTION FRACTION
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