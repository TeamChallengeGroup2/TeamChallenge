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
import random

#---------------------------------------------------------------------------------------------------
# LOADING THE DATA

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
                im_ESgt=sitk.ReadImage('Data\{}\{}_frame0{}_gt.nii.gz'.format(name,name,s))
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
    
    
#output=list with the patient number, the slices number, and the four 2D slices as arrays
slice_count=[0]
output=[]
for i in range(100):
    n=l[i][1]
    for h in range(n):
        output.append([l[i][0],h+1,l[i][2][h],l[i][3][h],l[i][4][h],l[i][5][h]])
        slice_count.append(slice_count[i]+n)
        

for i in range(951):
    if output[i][2].shape[0]<512:    #the bigger images have size 512x428
        L0=512-output[i][2].shape[0]
        L1=428-output[i][2].shape[1]
        output[i][2]=np.pad(output[i][2], ((0,L0),(0,L1)), 'constant', constant_values=0)
        output[i][3]=np.pad(output[i][3], ((0,L0),(0,L1)), 'constant', constant_values=0)
        output[i][4]=np.pad(output[i][4], ((0,L0),(0,L1)), 'constant', constant_values=0)
        output[i][5]=np.pad(output[i][5], ((0,L0),(0,L1)), 'constant', constant_values=0)

      

#---------------------------------------------------------------------------------------------------
# DEFINING THE NEURAL NETWORK
        
def buildUnet():

    cnn = keras.models.Sequential()

    layer0 = keras.layers.Conv2D(64, (3, 3), activation='relu', strides=1, input_shape=(32, 32, 1))
    cnn.add(layer0)

    layer1 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
    cnn.add(layer1)

    layer2 = keras.layers.Conv2D(128, (3, 3), activation='relu', strides=1)
    cnn.add(layer2)

    layer3 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
    cnn.add(layer3)

    layer4 = keras.layers.Conv2D(64, (3, 3), activation='relu', strides=1)
    cnn.add(layer4)

    layer5 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
    cnn.add(layer5)

    layer6 = keras.layers.Flatten() 
    cnn.add(layer6)

    layer7 = keras.layers.Dense(2, activation='softmax')
    cnn.add(layer7)

    sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.00005, nesterov=False)
    cnn.compile(loss='categorical_crossentropy', optimizer=sgd)
    
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
networkpath = r'Networks\trainednetwork.h5'
minibatches = 80
minibatchsize = 50 
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
 
# Training the network
trainingsamples=np.where(Train_labels==3)

# Initialise the network
cnn = buildUnet()

# Train the network
if trainnetwork:
    losslist = []
    t0 = time.time()

    for i in range(minibatches):
        # Take random trainingsamples and make the patches
        batch = random.sample(list(range(len(trainingsamples[0]))), int(minibatchsize/2))
        X, Y = make2Dpatches(trainingsamples,batch,frames,patchsize,1)
        # Train the network and compute the loss
        loss = cnn.train_on_batch(X,Y)
        losslist.append(loss)
        print('Batch: {}'.format(i))
        print('Loss: {}'.format(loss))

    # Save the network
    cnn.save(networkpath)
    
else:
    # Load the network
    cnn = keras.models.load_model(networkpath)
    
# Validation
validlosslist = []
probimage = np.zeros(Valid_frames.shape)

# Loop through all frames in the validation set
for j in range(np.shape(Valid_frames)[0]):
    
    validsamples=np.nonzero(Valid_labels[j])

    probabilities = np.empty((0,))
        
    minibatchsize = 100 # Can be set as large as the memory allows
    
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
        probimage[validsamples[0][m],validsamples[1][m]] = probabilities[m]

# Plot the loss and validation loss         
plt.close('all')
plt.figure()
plt.plot(losslist)  
plt.plot(validlosslist)

