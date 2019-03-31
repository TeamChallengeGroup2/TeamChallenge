"""
Creating the 2D patches

Team Challenge (TU/e & UU)
Team 2
"""

import numpy as np
   
def make2Dpatches(samples, batch, images, patchsize, label):
    # This function makes the patches given the trainingsamples, batch, frames 
    # used for the training, patchsize and label
    
    halfsize = int(patchsize/2)
    
    X = np.empty([len(batch),patchsize,patchsize,1],dtype=np.float32)
    Y = np.zeros((len(batch),2),dtype=np.int16) 
        
    for i in range(len(batch)):
        
        patch = images[samples[0][batch[i]],(samples[1][batch[i]]-halfsize):(samples[1][batch[i]]+halfsize),(samples[2][batch[i]]-halfsize):(samples[2][batch[i]]+halfsize)]
       
        X[i,:,:,0] = patch
        Y[i,label] = 1 
           
    return X, Y

def make2Dpatchestest(samples, batch, image, patchsize):
    # This function makes the patches given the validationsamples, batch, frames 
    # used for the validation and patchsize
    
    halfsize = int(patchsize/2)
    
    X = np.empty([len(batch),patchsize,patchsize,1],dtype=np.float32)
             
    for i in range(len(batch)):
        
        patch = image[(samples[0][batch[i]]-halfsize):(samples[0][batch[i]]+halfsize),(samples[1][batch[i]]-halfsize):(samples[1][batch[i]]+halfsize)]

        X[i,:,:,0] = patch  
        
    return X