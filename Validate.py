"""
Functions to plot some results and compute some validation metrics

Team Challenge (TU/e & UU)
Team 2
"""

import matplotlib.pyplot as plt
import numpy as np

# Function to plot some validation results
def plotResults(Test_frames, Test_labels, mask):
    
    plt.figure()
    for i in range(1, len(Test_frames), 3):
        plt.subplot(len(Test_frames), 3, i)
        plt.imshow(Test_frames[i-1], cmap='gray')
        plt.subplot(len(Test_frames), 3, i+1)
        plt.imshow(Test_frames[i], cmap='gray')
        plt.subplot(len(Test_frames), 3, i+2)
        plt.imshow(Test_frames[i+1], cmap='gray')

def metrics(mask, Test_labels):
    Dice = []
    Accuracy = []
    Sensitivity = []
    Specificity = []
    
    for i in range(mask.shape[0]):
        TP = np.sum(mask[i,:,:] * Test_labels[i,:,:])
        FN = np.sum(Test_labels[i,:,:]) - TP # P = TP + FN
        FP = len(np.where(mask[i,:,:] - Test_labels[i,:,:] == 1)[0])
        TN = len(np.where(mask[i,:,:] + Test_labels[i,:,:] == 0)[0])
    
        # Overcome computation problems (dividing by zero) by setting the number
        # of true positives and true negatives to a very small number if these 
        # numbers are originally equal to zero
        if TP == 0:
            TP = 0.0001
        if TN == 0:
            TN = 0.0001
            
        # Dice score
        DSC = 2*TP / (2*TP+FP+FN) 
        DSC = round(DSC, 4)
        Dice.append(DSC)  
          
        # Accuracy
        Acc = (TP+TN) / (TP+FN+FP+TN)
        Acc = round(Acc, 4)
        Accuracy.append(Acc)

        # Sensitivity and Specificity
        Sens = TP / (TP+FN)
        Sens = round(Sens, 4)
        Sensitivity.append(Sens)

        Spec = TN / (FP+TN)
        Spec = round(Spec, 4)
        Specificity.append(Spec)
        
    return Dice, Accuracy, Sensitivity, Specificity
