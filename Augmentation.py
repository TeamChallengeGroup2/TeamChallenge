# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 15:42:19 2019

Team Challenge (TU/e & UU)
Team 2
"""

import numpy as np
import cv2

# DATA AUGMENTATION
 
def augmentation(data,k):
     augmented_data=[]
     augmented_slices=[]
     #sl_count=[0]
 
     for i in range(k):
        
        # Extract ED and ES frame for each patient with the ground truths
        ED_frame=data[i][2]
        EDgt_frame=data[i][3]
        ES_frame=data[i][4]
        ESgt_frame=data[i][5]
        
        # Extract the slice number
        n=data[i][1]
        ED_frame_aug=ED_frame
        ED_frame_gt=EDgt_frame
        ES_frame_aug=ES_frame
        ES_frame_gt=ESgt_frame
        
        # Extract slices of ED, ES and ground truths
        for j in range(n):
            ED_slice=data[i][2][j]
            EDgt_slice=data[i][3][j]
            ES_slice=data[i][4][j]
            ESgt_slice=data[i][5][j]
            
            # Flip the slices
            if i<=k/2:
                ED_aug=np.fliplr(ED_slice)
                EDgt_aug=np.fliplr(EDgt_slice)
                ES_aug=np.fliplr(ES_slice)
                ESgt_aug=np.fliplr(ESgt_slice)
                #augmented_slices.append(ED_aug)
                
            # Apply Gaussian blur filter
            if i>k/2 and i<=k:
                ED_aug=cv2.GaussianBlur(ED_slice,(5,5),0) 
                EDgt_aug=EDgt_slice
                ES_aug=cv2.GaussianBlur(ES_slice,(5,5),0)
                ESgt_aug=ESgt_slice
                #augmented_slices.append(ED_aug)
                    
            # Make frames from slices (2D->3D)
            ED_frame_aug[j]=ED_aug
            ED_frame_gt[j]=EDgt_aug
            ES_frame_aug[j]=ES_aug
            ES_frame_gt[j]=ESgt_aug
                   
            # Save in list: patient number, slice number, ED, ground truth ED, ES, ground truth ES
            augmented_data.append([data[i][0]+100,j+1,ED_frame_aug,ED_frame_gt,ES_frame_aug,ES_frame_gt,data[i][6]])
            # sl_count.append(sl_count[i]+n)
                    
     return augmented_data

