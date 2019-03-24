import numpy as np
import cv2
from scipy.spatial.distance import directed_hausdorff

Valid_labels=np.load('Valid_labels.npy')
mask=np.load('mask.npy')

Valid_labels[Valid_labels==1]=0
Valid_labels[Valid_labels==2]=0
Valid_labels[Valid_labels==3]=1

distances=[]
for i in range(len(Valid_labels)):
    coordsValid=[]
    coordsMask=[]
    sobel_horizontalMask = cv2.Sobel(mask[i],cv2.CV_64F,1,0,ksize = 5)
    sobel_verticalMask = cv2.Sobel(mask[i],cv2.CV_64F,0,1,ksize=5)
    sobeltotalMask=sobel_horizontalMask+sobel_verticalMask
    
    sobel_horizontalValid = cv2.Sobel(Valid_labels[i],cv2.CV_64F,1,0,ksize = 5)
    sobel_verticalValid = cv2.Sobel(Valid_labels[i],cv2.CV_64F,0,1,ksize=5)
    sobeltotalValid=sobel_horizontalValid+sobel_verticalValid
    
    yValid,xValid = np.where(Valid_labels[i]!=0)
    yMask,xMask = np.where(mask[i]!=0)
    for i in range(len(yValid)):
        coordMask = ([xMask[i],yMask[i]])
        coordValid = ([xValid[i],yValid[i]])
        coordsMask.append(coordMask)
        coordsValid.append(coordValid)
        
print('Hausdorff calculated')
