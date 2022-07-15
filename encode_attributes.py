# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 15:40:00 2022

@author: Maria Eduarda
"""

# Imports
import numpy as np



# Function: Encode attributes
def encode_attributes(givenData: str, isFile:bool = True): 
    
    if isFile:
        # Open file
        file = open(givenData)
    
        # Load lesion type
        lesion_type = file.readline()
        
        # Load pathology (not necessary)
        _ = file.readline()
        
        # Load assessment (not necessary)
        _ = int(file.readline())
    
        # Load subtlety
        subtlety = int(file.readline())
    
        # Load information
        info = file.readline()
        
    # as used in flexible_dataset.py
    else:
        lesion_type = givenData.upper()
        info = givenData.upper()
        subtlety = 0

    stAttr = np.array([' N/A ', ' ARCHITECTURAL_DISTORTION', ' ASYMMETRIC_BREAST_TISSUE', 
                       ' FOCAL_ASYMMETRIC_DENSITY', ' IRREGULAR', ' LYMPH_NODE',
                       ' LOBULATED', ' OVAL', ' ROUND', ' TUBULAR'])
    ndAttr = np.array(['N/A\n', 'CIRCUMSCRIBED', 'ILL_DEFINED',
                        'MICROLOBULATED', 'OBSCURED', 'SPICULATED'])
    rdAttr = np.array([' N/A ', ' AMORPHOUS ', ' COARSE ',
                           ' DYSTROPHIC ', ' EGGSHELL ', ' FINE_LINEAR_BRANCHING ',
                           ' LARGE_RODLIKE ', ' LUCENT_CENTER ', ' LUCENT_CENTERED ',
                           ' MILK_OF_CALCIUM ', ' PLEOMORPHIC ', ' PUNCTATE ',
                           ' ROUND_AND_REGULAR ', ' SKIN ', ' VASCULAR '])
    thAttr = np.array(['N/A\n', 'CLUSTERED', 'DIFFUSELY_SCATTERED',
                           'LINEAR', 'REGIONAL', 'SEGMENTAL'])
    
    
    if lesion_type.find("MASS") != -1 and lesion_type.find("CALCIFICATION") != -1:
        patchRdAttr = ''
        rdAttrPosition = list(map(info.find, rdAttr))
        
        if max(rdAttrPosition) == -1:
            patchRdAttr = '0'
        while max(rdAttrPosition) != -1:
            index = rdAttrPosition.index(max(rdAttrPosition))
            if len(patchRdAttr) == 0:
                patchRdAttr += str(index)
            else:
                patchRdAttr += '.' + str(index)
            rdAttrPosition[index] = -1
        
        patchThAttr = ''
        thAttrPosition = list(map(info.find, thAttr))
        if max(thAttrPosition) == -1:
            patchThAttr = '0'
        while max(thAttrPosition) != -1:
            index = thAttrPosition.index(max(thAttrPosition))
            if len(patchThAttr) == 0:
                patchThAttr += str(index)
            else:
                patchThAttr += '.' + str(index)
            thAttrPosition[index] = -1
    
    # calcification file only have 2 attributes: there is no need to have two
    # zeros coded in the .csv
    elif lesion_type.find("CALCIFICATION") != -1:
        stAttr = rdAttr
        ndAttr = thAttr
    
    patchStAttr = ''
    stAttrPosition = list(map(info.find, stAttr)) 
    
    if max(stAttrPosition) == -1:
            patchStAttr = '0'
    while max(stAttrPosition) != -1:
        index = stAttrPosition.index(max(stAttrPosition))
        if len(patchStAttr) == 0:
            patchStAttr += str(index)
        else:
            patchStAttr += '.' + str(index)
        stAttrPosition[index] = -1 
    
    patchNdAttr = ''
    ndAttrPosition = list(map(info.find, ndAttr))
    
    if max(ndAttrPosition) == -1:
            patchNdAttr = '0'
    
    while max(ndAttrPosition) != -1:
        index = ndAttrPosition.index(max(ndAttrPosition))
        if len(patchNdAttr) == 0:
            patchNdAttr += str(index)
        else:
            patchNdAttr += '.' + str(index)
        ndAttrPosition[index] = -1

    if lesion_type.find("MASS") != -1 and lesion_type.find("CALCIFICATION") != -1:
        return np.array([patchStAttr, patchNdAttr, patchRdAttr, patchThAttr, subtlety])

    
    return np.array([patchStAttr, patchNdAttr, subtlety])



# Test
if __name__ == "__main__":
    
    # Define data path
    SOURCE = 'D:/Documentos/GitHub/breast-cancer-classification'
    PATH = '/data/ddsm/train//'
    FILE = 'benign_02_1280_LEFT_MLO_0'
    FILENAME = SOURCE + PATH + FILE + '.txt'

    # Expected result: [[10] [1] 2]
    print(encode_attributes(givenData=FILENAME))
    
    FILE = 'benign_01_0029_LEFT_CC_0'
    FILENAME = SOURCE + PATH + FILE + '.txt'

    # Expected result: [[7] [2] 3]
    print(encode_attributes(givenData=FILENAME))
    
    FILE = 'cancer_15_3399_RIGHT_CC_0'
    FILENAME = SOURCE + PATH + FILE + '.txt'

    # Expected result: [[1] [2, 5] [0] [0] 3]
    print(encode_attributes(givenData=FILENAME))
