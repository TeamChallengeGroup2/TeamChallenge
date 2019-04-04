"""
Computing the Ejection Fraction (EF)

Team Challenge (TU/e & UU)
Team 2
"""

import numpy as np

def EjectionFraction(maskED,maskES,voxelvolume):
    
    # Compute the stroke volume from the end-diastolic (ED) and end-systolic (ES) volume
    maxn=maskED.max()
    maskED=np.where(maskED>=(maxn-0.2),1,0)
    maskES=np.where(maskES>=(maxn-0.2),1,0)
    ED_volume = (np.sum(maskED))*voxelvolume
    ES_volume = (np.sum(maskES))*voxelvolume
    strokevolume = ED_volume - ES_volume
    
    # Compute the Ejection fraction
    LV_EF = (strokevolume/ED_volume)*100
    return LV_EF
