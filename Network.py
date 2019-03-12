# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:52:14 2019

Team Challenge (TU/e & UU)
Team 2
"""

import keras

def buildUnet():
    # This function defines the layers of the Unet ending with defining the optimizer
    
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
