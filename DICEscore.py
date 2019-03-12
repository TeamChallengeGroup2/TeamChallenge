import mumpy as np


def DSC(val_set, threshold, groundtruth):
    All_TP = []
    All_FP = []
    All_FN = []
    All_DSC = []
    
    # Loop trhough every validation image for this network
    for im_nr in range(len(val_set)):
        this_prob = np.load( 'Probimages'+ val_set[im_nr] + '.npy')
        
    TP = 0      # true positives
    TN = 0      # true negatives
    FP = 0      # false positives
    FN = 0      # false negatives
        
    # Loop through every pixel
    for i in range(this_prob.shape[0]):
        for j in range(this_prob.shape[1]):
            # define every pixel as a TP/FP/TN/FN and add it to the right category
            if this_prob[i,j] >=threshold and groundtruth[im_nr,i,j] !=0:
                TP += 1
            elif this_prob[i,j] >=threshold and groundtruth[im_nr,i,j] ==0:
                FP += 1
            elif this_prob[i,j] < threshold and groundtruth[im_nr,i,j] ==0:
                TN += 1
            elif this_prob[i,j] <threshold and groundtruth[im_nr,i,j] !=0:
                FN += 1
                
            
    All_TP.append(TP)
    All_FP.append(FP)
    All_FN.append(FN)
                
    DSC=2*TP[0]/(2*TP[0]+FP[0]+FN[0])
    
    All_DSC.append(DSC)
            
    print('DSC scores calculated')
                            
    return DSC
    