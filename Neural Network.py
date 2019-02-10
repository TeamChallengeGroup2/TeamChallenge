# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 15:51:08 2019

@author: s144314
"""

import keras

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