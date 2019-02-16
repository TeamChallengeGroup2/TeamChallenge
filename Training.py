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
                im_EDgt=sitk.ReadImage('Data\{}\{}_frame0{}.nii.gz'.format(name,name,s))
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
    
    
#output=list with the patient number, the slices number, and the four 2D slices as arrays
slice_count=[0]
output=[]
for i in range(100):
    n=l[i][1]
    for h in range(n):
        output.append([l[i][0],h+1,l[i][2][h],l[i][3][h],l[i][4][h],l[i][5][h]])
        slice_count.append(slice_count[i]+n)

#---------------------------------------------------------------------------------------------------
# DEFINING THE NEURAL NETWORK
        
def buildUnet():
    # This network is based on two papers of Ronneberger et al. (2015) and 
    # Poudel et al. (2016). The difference is the number of feature maps, which is
    # 64 and 32 at layer0 for the papers mentioned above, respectively. On the other hand,
    # the feature maps are in both papers doubled in each layer in the first phase and
    # after that halved.
    
    cnn = keras.models.Sequential()

    # The network starts with two 3x3 convolution layers, each followed with a rectified linear unit (ReLU)
    layer0 = keras.layers.Conv2D(64, (3, 3), activation='relu', strides=1, input_shape=(572, 572, 1))
    cnn.add(layer0)
    
    layer1 = keras.layers.Conv2D(64, (3, 3), activation='relu', strides=1)
    cnn.add(layer1)

    # Next, a 2x2 max pooling operation with stride 2 for downsampling is added
    layer2 = keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)
    cnn.add(layer2)

    # These steps of convolution and max pooling is repeated in the following layers resulting
    # in an output of 1024 feature maps
    layer3 = keras.layers.Conv2D(128, (3,3), activation='relu', strides=1)
    cnn.add(layer3)
    
    layer4 = keras.layers.Conv2D(128, (3,3), activation='relu', strides=1)
    cnn.add(layer4)
    
    layer5 = keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)
    cnn.add(layer5)
    
    layer6 = keras.layers.Conv2D(256, (3,3), activation='relu', strides=1)
    cnn.add(layer6)

    layer7 = keras.layers.Conv2D(256, (3,3), activation='relu', strides=1)
    cnn.add(layer7)

    layer8 = keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)
    cnn.add(layer8)

    layer9 = keras.layers.Conv2D(512, (3,3), activation='relu', strides=1)
    cnn.add(layer9)

    layer10 = keras.layers.Conv2D(512, (3,3), activation='relu', strides=1)
    cnn.add(layer10)

    layer11 = keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)
    cnn.add(layer11)

    layer12 = keras.layers.Conv2D(1024, (3,3), activation='relu', strides=1)
    cnn.add(layer12)

    layer13 = keras.layers.Conv2D(1024, (3,3), activation='relu', strides=1)
    cnn.add(layer13)
    
    # Next, the upsampling is performed by a 2x2 convolution and after that two times
    # a 3x3 convolution followed by a rectified linear unit (ReLU)
    layer14 = keras.layers.Conv2D(1024, (2,2), activation='relu', strides=1)
    cnn.add(layer14)
    
    layer15 = keras.layers.Conv2D(512, (3,3), activation='relu', strides=1)
    cnn.add(layer15)
    
    layer16 = keras.layers.Conv2D(512, (3,3), activation='relu', strides=1)
    cnn.add(layer16)
    
    layer17 = keras.layers.Conv2D(512, (2,2), activation='relu', strides=1)
    cnn.add(layer17)
    
    layer18 = keras.layers.Conv2D(256, (3,3), activation='relu', strides=1)
    cnn.add(layer18)
    
    layer19 = keras.layers.Conv2D(256, (3,3), activation='relu', strides=1)
    cnn.add(layer19)
    
    layer20 = keras.layers.Conv2D(256, (2,2), activation='relu', strides=1)
    cnn.add(layer20)
    
    layer21 = keras.layers.Conv2D(128, (3,3), activation='relu', strides=1)
    cnn.add(layer21)
    
    layer22 = keras.layers.Conv2D(128, (3,3), activation='relu', strides=1)
    cnn.add(layer22)
    
    layer23 = keras.layers.Conv2D(128, (2,2), activation='relu', strides=1)
    cnn.add(layer23)
    
    layer24 = keras.layers.Conv2D(64, (3,3), activation='relu', strides=1)
    cnn.add(layer24)
    
    layer24 = keras.layers.Conv2D(64, (3,3), activation='relu', strides=1)
    cnn.add(layer24)
    
    # Then, a 1x1 convolution layer is added to map each feature vector to the desired
    # number of classes, so 1 or 0 for pixels included or excluded as left ventricle
    layer25 = keras.layers.Conv2D(2, (1,1), activation='relu', strides=1)
    cnn.add(layer25)
    
    # The tensor is flattened
    layer26 = keras.layers.Flatten() 
    cnn.add(layer26)

    # Finally the network is optimized using the stochastic gradient descent optimizer
    keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.00005, nesterov=False)
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
     
# Shuffle the data to take a random subset for training later
random.shuffle(l)

# Split the list l into a list containing all ED frames
EDframes = []
EDground = []
ESframes = []
ESground = []

for i in range(len(l)):
    EDframes.append(l[i][2])
    EDground.append(l[i][3])
    ESframes.append(l[i][4])
    ESground.append(l[i][5])

# Take the ESframes and ED frames 
frames = ESframes+EDframes
groundtruth = ESground+EDground

Train_frames = frames[:int(len(frames)/2)]
Valid_frames = frames[int(len(frames)/2):int(len(frames)-len(frames)/4)]
Test_frames = frames[int(len(frames)-len(frames)/4):]

Train_labels = groundtruth[:int(len(groundtruth)/2)]
Valid_labels = groundtruth[int(len(groundtruth)/2):int(len(groundtruth)-len(groundtruth)/4)]
Test_labels = groundtruth[int(len(groundtruth)-len(groundtruth)/4):]
    
# Training the network
trainingsamples=np.nonzero(Train_labels)
validsamples=np.nonzero(Valid_labels)

# Initialise the network
cnn = buildUnet()

minibatches = 80
minibatchsize = 50 

losslist = []
t0 = time.time()

for i in range(minibatches):
    # Select random training en validation samples and perform training and validation steps here.  
    batch = random.sample(list(range(len(trainingsamples[0]))), int(minibatchsize))
    
    X, Y = make2Dpatches(trainingsamples,batch,frames,32,1)
    loss = cnn.train_on_batch(X,Y)
    losslist.append(loss)
    print('Batch: {}'.format(i))
    print('Loss: {}'.format(loss))
   
# Plot the loss function
plt.close('all')
plt.figure()
plt.plot(losslist)  

""" Validation (when training is finished and optimized)

# Show the time needed to train the network
t1 = time.time()
print('Training time: {} seconds'.format(t1-t0))

#  Validate the trained network on the images that were left out during training 
validlosslist = []

for i in range(0,len(validsamples[0]),minibatchsize):
    print('{}/{} samples labelled'.format(i,len(validsamples[0])))
    
    if i+minibatchsize < len(validsamples[0]):
        valbatch = np.arange(i,i+minibatchsize)        
    else:
        valbatch = np.arange(i,len(validsamples[0]))        
    
    # Create 2D patches        
    Xval = make2Dpatchestest(validsamples,valbatch,groundtruth,patchsize)
    
    # Compute the predictions of which pixels are left ventricle and which not                
    prob = cnn.predict(Xval, batch_size=minibatchsize)
    probabilities = np.concatenate((probabilities,prob[:,1]))
   
# Compute the loss         
for j in range(minibatches):
    validloss = cnn.test_on_batch(Xval,prob)
    validlosslist.append(validloss)
      
# Create the probability image of the segmented left ventricles      
for i in range(len(valsamples[0])):
    probimage[valsamples[0][i],valsamples[1][i]] = probabilities[i]

plt.close('all')
plt.figure()
plt.plot(losslist)  
plt.plot(validlosslist)
"""
