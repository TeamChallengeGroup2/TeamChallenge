"""
Main script
Team Challenge (TU/e & UU)
Team 2
"""

import numpy as np
import time
import random
import os
import cv2
import pickle

from Data import loadData
from Cropping import cropImage
from fcnNetwork import fcn_model
from Validate import plotResults, metrics
from DataAugmentation import create_Augmented_Data
from EF_calculation import EjectionFraction

# -----------------------------------------------------------------------------
# INPUT
path = os.path.realpath("Main.py").replace("\\Main.py","")
networkpath = r'trainednetwork.h5'
nr_of_batches_augmentation = 30
batchsize_augmentation = 25
batchsize = 5
epochs = 20
augmentation = True
trainnetwork = True
testing = True
plot = False
Compute_EjectionFraction = True

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

# -----------------------------------------------------------------------------
# CROPPING
Train_slices_cropped, excluded_Train = cropImage(Train_frames)
Test_slices_cropped, excluded_Test = cropImage(Test_frames)

# -----------------------------------------------------------------------------
# DIVIDING THE DATA

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

# -----------------------------------------------------------------------------
# DATA AUGMENTATION
if augmentation:
    slicesTrain, groundtruthTrain = create_Augmented_Data(slicesTrain, groundtruthTrain, batchsize_augmentation, nr_of_batches_augmentation)

# -----------------------------------------------------------------------------
# TRAINING
# Initialize the model
cnn  = fcn_model((126,126,1),2,weights=None)

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
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(hist.history, f)
    
else:
    # Load the network
    cnn.load_weights(networkpath)
    
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

if Compute_EjectionFraction:
    print ('Start computing the Ejection Fraction')
    
    EF = []
    EF_gt = []
    count = 0
    
    for k in range(len(Test_frames_info)):
        
        # Only compute the EF if all frames of a patient can be cropped
        if Test_frames[k][0] in excluded_Test:
            LV_EF = 0
            
        else:
            # Determine the voxelvolume
            spacing = Test_frames_info[k][2]
            voxelvolume = spacing[0] * spacing[1] * spacing[2]
            maskED = mask[count:count+Test_frames_info[k][1]]
            maskES = mask[count+Test_frames_info[k][1]:count+2*Test_frames_info[k][1]]
            count = count + 2 * Test_frames_info[k][1]
            
            # Only compute the EF if both the ED and ES frame are predicted
            if len(maskED) != 0 and len(maskES) != 0:
                # Compute the Ejection fraction per patient
                LV_EF = EjectionFraction(maskED,maskES,voxelvolume)
            
            else:
                LV_EF = 0
            
            # Save all EF in a list
            EF.append(LV_EF)
        
        # Compute the EF for the ground truth and save in a list
        gtED = Test_frames[k][3]
        gtES = Test_frames[k][5]
        LV_EF_gt = EjectionFraction(gtED,gtES,voxelvolume)
        EF_gt.append(LV_EF_gt)
        
    print ('Ejection Fraction is computed')
