# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 16:16:51 2019

@author: s144314
"""

#!/usr/bin/env python
# coding: utf-8
"""
This code gives as output the slices of end-diastolic and end-systolic frames of 100 patients, including both the initial and segmented images.
To know how many slices (N) the patient number X has, use:
N = slice_count[X] - slice_count[X-1]
To get his slice number n (n being a number between 1 and N, including both), use:
Y = output[slice_count[X-1] + n-1]
Then:
Y[0] = patient number
Y[1] = slice number
Y[2] = ED 2D original image
Y[3] = ED 2D ground truth
Y[4] = ES 2D original image
Y[5] = ES 2D ground truth
"""

import os
import numpy as np
import SimpleITK as sitk
import keras
import time
import matplotlib.pyplot as plt
import cv2
import math 
import random

#---------------------------------------------------------------------------------------------------
# LOADING THE DATA

def loadData():
    l=[]
    for i, name in enumerate(os.listdir('Data')):
        data = open('Data\{}\Info.cfg'.format(name), 'r')
        
        ED=data.readline()    #end-diastolic frame information
        for s in ED.split():
            if s.isdigit():   #end-diastolic frame number
                #reading the end-diastolic 3d images:
                if int(s)<10:
                    im_EDframe= sitk.ReadImage('Data\{}\{}_frame0{}.nii.gz'.format(name,name,s))
                    im_EDgt=sitk.ReadImage('Data\{}\{}_frame0{}_gt.nii.gz'.format(name,name,s))
                else:
                    im_EDframe= sitk.ReadImage('Data\{}\{}_frame{}.nii.gz'.format(name,name,s))
                    im_EDgt=sitk.ReadImage('Data\{}\{}_frame{}_gt.nii.gz'.format(name,name,s))
                    
        ES=data.readline()    #end-systolic frame information
        for s in ES.split():
            if s.isdigit():   #end-systolic frame number
                #reading the end-systolic 3d images:
                if int(s)<10:
                    im_ESframe= sitk.ReadImage('Data\{}\{}_frame0{}.nii.gz'.format(name,name,s))
                    im_ESgt=sitk.ReadImage('Data\{}\{}_frame0{}.nii.gz'.format(name,name,s))
                else:
                    im_ESframe= sitk.ReadImage('Data\{}\{}_frame{}.nii.gz'.format(name,name,s))
                    im_ESgt=sitk.ReadImage('Data\{}\{}_frame{}_gt.nii.gz'.format(name,name,s))
                    
        #Converting the 3d images into 3 dimensional arrays:        
        arr_EDframe= sitk.GetArrayFromImage(im_EDframe)
        arr_EDgt= sitk.GetArrayFromImage(im_EDgt)
        arr_ESframe= sitk.GetArrayFromImage(im_ESframe)
        arr_ESgt= sitk.GetArrayFromImage(im_ESgt)
        
        NSlices=arr_EDframe.shape[0]
        
        #l=list with the patient number, the number of slices per frame, and the four 3D frames as arrays
        l.append([i+1, NSlices,arr_EDframe,arr_EDgt,arr_ESframe,arr_ESgt])
        
    return l

#---------------------------------------------------------------------------------------------------
# CROPPING THE IMAGES

def cropROI(arrayin, slicenr):
    #input: an array with indices [slicenr, imageX, imageY]
    # slicenr: averaging over this number of slices
    
    LVradius=20 #the radius used in frst, which should be the radius of the LV in the top image
    cropdiam=63 #the length in X and Y direction of the cropped image
    multi_mindist = []
    multi_circles = []
    sumdist1 = 0
    sumdist2 = 0
    
    for i in range(slicenr):
        topslice=arrayin[i,:,:]
        center=[topslice.shape[1]/2,topslice.shape[0]/2]        #find coordinates of center of image

        im = np.array((topslice/np.amax(topslice)) * 255, dtype = np.uint8)
    
        circles=cv2.HoughCircles(im, cv2.HOUGH_GRADIENT, 1, 70, param1=25, param2=20, minRadius=LVradius-14, maxRadius=LVradius+10)
        multi_circles.append(circles)

    #function to calculate distance from coordinates of HoughCircles to center of the image
        def calculateDistance(x1,y1,x2,y2):  
            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
            return dist

    #calculate distances and add to list dist
        dist=[]
        for j in range(circles.shape[1]):
            d=calculateDistance(circles[0,j,0],circles[0,j,1],center[0],center[1])
            dist.append(d)

        mindist=np.argmin(dist)         #find index of minimal distance
        multi_mindist.append(mindist)
        sumdist1 += int(circles[0,mindist,1])
        sumdist2 += int(circles[0,mindist,0])

    sumdist1 = int(sumdist1/(slicenr))
    sumdist2 = int(sumdist2/(slicenr))
    
    #find the coordinates around which to crop
    cropcoor_x1=sumdist1-cropdiam
    cropcoor_x2=sumdist1+cropdiam
    cropcoor_y1=sumdist2-cropdiam
    cropcoor_y2=sumdist2+cropdiam
    
    croppedim=im[sumdist1-cropdiam:sumdist1+cropdiam,sumdist2-cropdiam:sumdist2+cropdiam]
      
    return croppedim, cropcoor_x1, cropcoor_x2, cropcoor_y1, cropcoor_y2

l=loadData()

#output=list with the patient number, the slices number, and the four 2D slices as arrays
output=[]
slice_count=[0]
for j in range(len(l)):
    patient1=l[j]
    
    EDpatient1=patient1[2]
    ESpatient1=patient1[4]
    
    cropped_EDim, EDx1, EDx2, EDy1, EDy2=cropROI(EDpatient1,4)
    cropped_ESim, ESx1, ESx2, ESy1, ESy2=cropROI(ESpatient1,4)
    if cropped_EDim.size and cropped_ESim.size: #crop only if HoughCircles is able to find a circle
        n=l[j][1]
        for h in range(n):
            EDslice=l[j][2][h]
            EDslicegt=l[j][3][h]
            ESslice=l[j][4][h]
            ESslicegt=l[j][5][h]
            
            output.append([l[j][0],h+1,EDslice[EDx1:EDx2, EDy1:EDy2],EDslicegt[EDx1:EDx2, EDy1:EDy2],ESslice[ESx1:ESx2, ESy1:ESy2],ESslicegt[ESx1:ESx2, ESy1:ESy2]])
            slice_count.append(slice_count[j]+n)
      
print('Images cropped')
#---------------------------------------------------------------------------------------------------
# DEFINING THE NEURAL NETWORK
        
def buildUnet():

    cnn = keras.models.Sequential()

    layer0 = keras.layers.Conv2D(64, (3, 3), activation='relu', strides=1)
    cnn.add(layer0)
    
    layer1 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
    cnn.add(layer1)

    layer2 = keras.layers.Conv2D(128, (3, 3), activation='relu', strides=1)
    cnn.add(layer2)
    
    layer3 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
    cnn.add(layer3)
    
    layer4 = keras.layers.Conv2D(512, (3, 3), activation='relu', strides=1)
    cnn.add(layer4)
    
    layer5 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
    cnn.add(layer5)

    layer6 = keras.layers.Flatten() 
    cnn.add(layer6)
     
    layer7 = keras.layers.Dense(120, activation='relu')
    cnn.add(layer7)
    
    layer8 = keras.layers.Dense(84, activation='relu')
    cnn.add(layer8)

    layer9 = keras.layers.Dense(2, activation='softmax')
    cnn.add(layer9)

    adam = keras.optimizers.adam(lr=0.0001)
    cnn.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
     
    return cnn

#-------------------------------------------------------------------------------------------------
# TRAINING THE NETWORK
    
def make2Dpatches(samples, batch, images, patchsize, label):
    
    halfsize = int(patchsize/2)
    
    X = np.empty([len(batch),patchsize,patchsize,1],dtype=np.float32)
    Y = np.zeros((len(batch),2),dtype=np.int16) 
        
    for i in range(len(batch)):
        
        patch = images[samples[0][batch[i]],(samples[1][batch[i]]-halfsize):(samples[1][batch[i]]+halfsize),(samples[2][batch[i]]-halfsize):(samples[2][batch[i]]+halfsize)]
       
        X[i,:,:,0] = patch
        Y[i,label] = 1 
           
    return X, Y

def make2Dpatchestest(samples, batch, image, patchsize):
    
    halfsize = int(patchsize/2)
    
    X = np.empty([len(batch),patchsize,patchsize,1],dtype=np.float32)
    Y = np.zeros((len(batch),2),dtype=np.int16)
             
    for i in range(len(batch)):
        
        patch = image[(samples[0][batch[i]]-halfsize):(samples[0][batch[i]]+halfsize),(samples[1][batch[i]]-halfsize):(samples[1][batch[i]]+halfsize)]

        X[i,:,:,0] = patch  
        
    return X
  
# Input
networkpath = r'trainednetwork.h5'
minibatches = 50
minibatchsize = 200
patchsize = 32
trainnetwork = True

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

print(frames[0].shape)
print(groundtruth[0].shape)

frames=np.array(frames)
groundtruth=np.array(groundtruth)


#pad the images with zeros to allow patch extraction at all locations
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

# Training the network
positivesamples = np.nonzero(Train_labels)
negativesamples = np.nonzero(Train_frames-Train_labels)

validsamples=np.where(Valid_labels==3)



# Train the network
if trainnetwork:
    trainlosslist = []
    validlosslist = []
    t0 = time.time()

    for i in range(minibatches):
#        # Take random trainingsamples and make the patches
        posbatch = random.sample(list(range(len(positivesamples[0]))),int(minibatchsize/2))
        negbatch = random.sample(list(range(len(negativesamples[0]))),int(minibatchsize/2))
        
        valid_batch = random.sample(list(range(len(validsamples[0]))), int(minibatchsize/2))
        
        Xpos, Ypos = make2Dpatches(positivesamples,posbatch,Train_frames,patchsize,1) # double patchsize for rotation
        Xneg, Yneg = make2Dpatches(negativesamples,negbatch,Train_frames,patchsize,0)   # it is cropped later
        
        x_valid, y_valid = make2Dpatches(validsamples, valid_batch, Valid_frames, patchsize, 1)
        
        Xtrain = np.vstack((Xpos,Xneg))
        Ytrain = np.vstack((Ypos,Yneg))

        train_loss = cnn.train_on_batch(Xtrain,Ytrain)
        valid_loss = cnn.test_on_batch(x_valid, y_valid)
        trainlosslist.append(train_loss)
        validlosslist.append(valid_loss)
        print('Batch: {}'.format(i))
        print('Train Loss: {} \t Train Accuracy: {}'.format(train_loss[0], train_loss[1]))
        print('Valid loss: {} \t Valid Accuracy: {}'.format(valid_loss[0], valid_loss[1]))
                
    # Save the network
    cnn.save(networkpath)
    t1 = time.time()
    print('\nTraining time: {} seconds'.format(t1 - t0))
else:
    # Load the network
    cnn = keras.models.load_model(networkpath)
'''
#---------------------------------------------------------------------------------    

# VALIDATION
validlosslist = []
probimage = np.zeros(Valid_frames.shape)

# Loop through all frames in the validation set
for j in range(np.shape(Valid_frames)[0]):
    
    # Take all labels of the Left ventricle (3) or all structures together
    validsamples=np.where(Valid_labels[j]==3)
    #validsamples=np.nonzero(Valid_labels[j])

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
    
    # Compute the loss for the validation        
    for l in range(minibatches):
        validloss = cnn.test_on_batch(Xval,prob)
        validlosslist.append(validloss)
    
    # Create the probability image        
    for m in range(len(validsamples[0])):
        probimage[j,validsamples[0][m],validsamples[1][m]] = probabilities[m]

# Plot the loss and validation loss         
plt.close('all')
plt.figure()
plt.plot(losslist)  
plt.plot(validlosslist)
'''