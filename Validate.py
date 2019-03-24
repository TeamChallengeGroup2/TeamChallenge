# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 14:51:13 2019

@author: s144314
"""

import matplotlib.pyplot as plt
import numpy as np

def plotResults(Valid_frames, Valid_labels, mask):
    
    plt.figure()
    for i in range(1, len(Valid_frames), 3):
        plt.subplot(len(Valid_frames), 3, i)
        plt.imshow(Valid_frames[i-1], cmap='gray')
        plt.subplot(len(Valid_frames), 3, i+1)
        plt.imshow(Valid_frames[i], cmap='gray')
        plt.subplot(len(Valid_frames), 3, i+2)
        plt.imshow(Valid_frames[i+1], cmap='gray')
        
def calculateDice(mask, Valid_labels):
    
    dices=[]
    
    for i in range(len(Valid_labels)):
        gt=np.where(Valid_labels[i]==3, 1, 0)
        dice=np.sum(mask[i,gt==1])*2.0/(np.sum(mask[i])+np.sum(gt))
        dices.append(dice)
    return dices

def metrics(mask,Valid_labels):
    Accuracy = []
    Sensitivity = []
    Specificity = []

    # Compute the True positives (TP), False negatives (FN), False positives (FP) and True negatives (TN)
    for i in range(mask.shape[0]):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for j in range(mask.shape[1]):
            for k in range(mask.shape[2]):
                if mask[i,j,k] == 1 and Valid_labels[i,j,k] == 3:
                    TP = TP+1
                    
                if mask[i,j,k] == 0 and Valid_labels[i,j,k] == 3:
                    FN = FN+1
                       
                if mask[i,j,k] == 1 and Valid_labels[i,j,k] != 3:
                    FP = FP+1
                            
                if mask[i,j,k] == 0 and Valid_labels[i,j,k] != 3:
                    TN = TN+1
                        
        # Accuracy
        Acc = (TP+TN)/(TP+FN+FP+TN)
        Accuracy.append(Acc)
   
        # Sensitivity and Specificity
        Sens = TP/(TP+FN)
        Sensitivity.append(Sens)
        Spec = TN/(FP+TN)
        Specificity.append(Spec)
        
    return Accuracy, Sensitivity, Specificity
