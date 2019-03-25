import numpy as np

def DSC(segmented, groundtruth):
    All_TP = []
    All_FP = []
    All_FN = []
    All_DSC = []
        
    TP = 0      # true positives
    TN = 0      # true negatives
    FP = 0      # false positives
    FN = 0      # false negatives
        
    # Loop through every pixel
    for im_nr in range(segmented.shape[0]):
        for i in range(segmented.shape[1]):
            for j in range(segmented.shape[2]):
                # define every pixel as a TP/FP/TN/FN and add it to the right category
                if segmented[im_nr,i,j] == 1 and groundtruth[im_nr,i,j] ==3:
                    TP += 1
                elif segmented[im_nr,i,j] == 1 and groundtruth[im_nr,i,j] !=3:
                    FP += 1
                elif segmented[im_nr,i,j] == 0 and groundtruth[im_nr,i,j] !=3:
                    TN += 1
                elif segmented[im_nr,i,j] == 0 and groundtruth[im_nr,i,j] ==3:
                    FN += 1
                
            
        All_TP.append(TP)
        All_FP.append(FP)
        All_FN.append(FN)
                    
        DSC=2*TP/(2*TP+FP+FN)
        
        All_DSC.append(DSC)
            
    print('DSC scores calculated')
                            
    return All_DSC
    
