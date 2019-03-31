"""
Functions to plot some results and compute some validation metrics

Team Challenge (TU/e & UU)
Team 2
"""

import matplotlib.pyplot as plt

# Function to plot some validation results
def plotResults(Valid_frames, Valid_labels, mask):
    
    plt.figure()
    for i in range(1, len(Valid_frames), 3):
        plt.subplot(len(Valid_frames), 3, i)
        plt.imshow(Valid_frames[i-1], cmap='gray')
        plt.subplot(len(Valid_frames), 3, i+1)
        plt.imshow(Valid_frames[i], cmap='gray')
        plt.subplot(len(Valid_frames), 3, i+2)
        plt.imshow(Valid_frames[i+1], cmap='gray')

def metrics(mask,Valid_labels):
    Dice = []
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
        
        # Overcome computation problems (dividing by zero) by setting the number
        # of true positives and true negatives to a very small number if these 
        # numbers are originally equal to zero
        if TP==0:
            TP=0.0001
        if TN == 0:
            TN = 0.0001
            
        # Dice score
        DSC=2*TP/(2*TP+FP+FN)   
        if TP == 0.0001:
            DSC = 0
        Dice.append(DSC)  
          
        # Accuracy
        Acc = (TP+TN)/(TP+FN+FP+TN)
        Accuracy.append(Acc)

        # Sensitivity and Specificity
        Sens = TP/(TP+FN)
        if TP == 0.0001:
            Sens = 0
        Sensitivity.append(Sens)

        Spec = TN/(FP+TN)
        if TN == 0.0001:
            Spec = 0
        Specificity.append(Spec)
        
    return Dice, Accuracy, Sensitivity, Specificity
