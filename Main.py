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
random.seed(0)
random.shuffle(data_original)

# Split up the data set into a training and test set. Note that the validation frames
# are included in the Train_frames array
Train_frames = data_original[:3*int(len(data_original)/4)]
Test_frames = data_original[int(len(data_original)-len(data_original)/4):]

#Info: per frame, patient number,number of slices and spacing
Test_frames_info=[]
for i in range(len(Test_frames)):
    Test_frames_info.append([Test_frames[i][0],Test_frames[i][1],Test_frames[i][6]])
    
    
# Shuffle the training data to take a random subset for training later
random.shuffle(Train_frames)

Train_slices_cropped = cropImage(Train_frames)
Test_slices_cropped = cropImage(Test_frames)

# Extract the frames and spacing
slicesTrain = []
groundtruthTrain = []
spacingsTrain = []

slicesTest = []
groundtruthTest = []
spacingsTest = []

for i in range(len(Train_slices_cropped)):    
    # Append the ED frame 
    slicesTrain.append(Train_slices_cropped[i][2])
    # Append the ES frame
    slicesTrain.append(Train_slices_cropped[i][4])
    # Append the ED groundtruth 
    groundtruthTrain.append(Train_slices_cropped[i][3])
    # Append the ES groundtruth
    groundtruthTrain.append(Train_slices_cropped[i][5])
    # Append the spacing
    spacingsTrain.append(Train_slices_cropped[i][6])
    
for i in range(len(Test_slices_cropped)):    
    # Append the ED frame 
    slicesTest.append(Test_slices_cropped[i][2])
    # Append the ES frame
    slicesTest.append(Test_slices_cropped[i][4])
    # Append the ED groundtruth 
    groundtruthTest.append(Test_slices_cropped[i][3])
    # Append the ES groundtruth
    groundtruthTest.append(Test_slices_cropped[i][5])
    # Append the spacing
    spacingsTest.append(Test_slices_cropped[i][6])

slicesTrain = np.array(slicesTrain)
slicesTest = np.array(slicesTest)

groundtruthTrain = np.array(groundtruthTrain)
groundtruthTest = np.array(groundtruthTest)

# Scale the masks to binary masks
groundtruthTrain = cv2.threshold(groundtruthTrain,2.5,1.0,cv2.THRESH_BINARY)[1]
groundtruthTest = cv2.threshold(groundtruthTest,2.5,1.0,cv2.THRESH_BINARY)[1]

# Augment the data
if augmentation:
    slicesTrain, groundtruthTrain = create_Augmented_Data(slicesTrain, groundtruthTrain, batchsize_augmentation, nr_of_batches_augmentation)

# Initialize the model
cnn  = fcn_model((128,128,1),2,weights=None)

# -----------------------------------------------------------------------------
# TRAINING
if trainnetwork:
    
    # Train the network
    print ('Start training')
    t0 = time.time()
    
    # Train the network 
    hist=cnn.fit(slicesTrain[:,:,:,np.newaxis], groundtruthTrain[:,:,:,np.newaxis],
            batch_size=batchsize, epochs=epochs, verbose=1, shuffle=True, validation_split = 0.3)
    
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
    mask = cnn.predict(slicesTest[:,:,:,np.newaxis], verbose=1)

    # As the prediction have the channels dimension (3th dimension per slice), 
    # to go back to 2 dimensions per slice:
    mask=np.squeeze(mask)
    
    # Compute the DICE coefficient, accuracy, sensitivity and specificity per image
    Dice, Accuracy, Sensitivity, Specificity = metrics(mask, groundtruthTest)
    
    # Plot the results
    if plot:
        plotResults(slicesTest, groundtruthTest, mask)

print ('Testing is finished')
    
# -----------------------------------------------------------------------------
# EJECTION FRACTION

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

if EjectionFraction:
    print ('Start computing the Ejection Fraction')
    
    EF=[]
    EF_gt=[]
    count=0
    for k in range(0,len(Test_frames_info)):
        # Determine the voxelvolume
        voxelvolume = Test_frames_info[k][2]
        maskED = mask[count:count+Test_frames_info[k][1]]
        maskES = mask[Test_frames_info[k][1]:2*Test_frames_info[k][1]]
        count=count+2*Test_frames_info[k][1]
        
        # Compute the Ejection fraction per patient and save in a list
        LV_EF = EjectionFraction(maskED,maskES,voxelvolume)
        EF.append(LV_EF)
        
        # Ground truth
        gtED = Test_frames[k][3]
        gtES = Test_frames[k][5]
        LV_EF_gt = EjectionFraction(gtED,gtES,voxelvolume)
        EF_gt.append(LV_EF_gt)
        
        
        
    print ('Ejection Fraction is computed')
