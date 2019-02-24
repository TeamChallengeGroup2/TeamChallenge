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
import pickle
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
    
    #arr_EDframe=arr_EDframe[:,50:100,50:100]
    #arr_EDgt=arr_EDgt[:,50:100,50:100]
    #arr_ESframe=arr_ESframe[:,50:100,50:100]
    #arr_ESgt=arr_ESgt[:,50:100,50:100]
    
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
        
def buildLeNet():
    
    cnn = keras.models.Sequential()
    
    # 6 output filters in the convolution / number of kernels
    # (5,5) kernel size
    # 'relu' activation function
    # input shape
    # veel andere parameters mogelijk
    layer0 = keras.layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 1))
    cnn.add(layer0)
    print(layer0.input_shape)
    print(layer0.output_shape)
    
    layer1 = keras.layers.MaxPooling2D(pool_size=(2, 2))
    cnn.add(layer1)
    print(layer1.output_shape)
    
    layer2 = keras.layers.Conv2D(16, (5, 5), activation='tanh')
    cnn.add(layer2)
    print(layer2.output_shape)
    
    layer3 = keras.layers.MaxPooling2D(pool_size=(2, 2))
    cnn.add(layer3)
    print(layer3.output_shape)
    
    layer4 = keras.layers.Flatten() 
    cnn.add(layer4)
    print(layer4.output_shape)
    
    # Output array of size 120 (neurons)
    layer5 = keras.layers.Dense(120, activation='tanh')
    cnn.add(layer5)
    print(layer5.output_shape)
    
    layer6 = keras.layers.Dense(84, activation='tanh')
    cnn.add(layer6)
    print(layer6.output_shape)
    
    layer7 = keras.layers.Dense(2, activation='softmax')
    cnn.add(layer7)
    print(layer7.output_shape)
    
    # Adam optimizer
    adam = keras.optimizers.adam(lr=0.001)
    cnn.compile(loss='categorical_crossentropy', optimizer=adam)
    
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

Train_frames = frames[:int(len(frames)/2)]
Valid_frames = frames[int(len(frames)/2):int(len(frames)-len(frames)/4)]
Test_frames = frames[int(len(frames)-len(frames)/4):]

Train_frames=np.array(Train_frames)
Valid_frames=np.array(Valid_frames)

Train_labels = groundtruth[:int(len(groundtruth)/2)]
Valid_labels = groundtruth[int(len(groundtruth)/2):int(len(groundtruth)-len(groundtruth)/4)]
Test_labels = groundtruth[int(len(groundtruth)-len(groundtruth)/4):]

Train_labels=np.array(Train_labels)
Valid_labels=np.array(Valid_labels)
 
# Training the network
trainingsamples=np.where(Train_labels==3)
validsamples=np.where(Valid_labels==3)

# Initialise the network
cnn = buildLeNet()

minibatches = 80
minibatchsize = 50 
patchsize = 32

losslist = []
t0 = time.time()

for i in range(minibatches):
    # Select random training en validation samples and perform training 
    batch = random.sample(list(range(len(trainingsamples[0]))), int(minibatchsize/2))
    X, Y = make2Dpatches(trainingsamples,batch,frames,32,1)
    loss = cnn.train_on_batch(X,Y)
    losslist.append(loss)
    print('Batch: {}'.format(i))
    print('Loss: {}'.format(loss))
   
# Plot the loss function
plt.close('all')
plt.figure()
plt.plot(losslist)  
