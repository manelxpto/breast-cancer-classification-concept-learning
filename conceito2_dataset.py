# -*- coding: utf-8 -*-
"""
Created on Thu May 12 10:27:52 2022

@author: Érica Gomes e Manuel Fortunato
"""

# Imports
import glob
import os
import sys
import tqdm
import numpy as np
import pandas as pd
from PIL import Image

# Project Imports
from auxiliary import get_mask_patch
from encode_attributes import encode_attributes




# Function: Create data set
def create_dataset(path=r'C:\Users\efgom\OneDrive\Documentos\GitHub\breast-cancer-classification'):
    # If something fails you may need to set your project path (uncomment if needed)
    # It is better to set PATH as the current working directory (to do this we use "os.getcwd()"")
    # path = os.getcwd()
    # path = 
    
    # Append current working directory to PATH to export stuff outside this folder
    if path not in sys.path:
        sys.path.append(path)
    
    
    
    # All the images should have the same size
    IMG_SIZE = 224
    
    
    # Array that will allow us to distiguish training, validation and test examples
    extraPath = np.array (['/train', '/val', '/test'])

    # Go through each example in the array
    for i in tqdm.tqdm(range(0, extraPath.size)):
        # is increased when a new image is considered in order to name the new 
        # files properly
        countMasses = 0

        
        # Dictionaries that will be used to export the .csv files
        yM = {'imgName': []}
        attrM = {'imgName': [],
                     'attr1': [],
                     'attr2': [],}
        
       
        # Path that leads to the raw data in each subset (train, val and test)
        rawPath = path + '/data/ddsm' + extraPath[i]
        # rawPath = 'data/ddsm' + extraPath[i]
        
        # Creates the path that will lead to the processed data in each subset
        procPathMasses = path + '/data/conceptl/conceito2_2/masses' + extraPath[i]
        # procPathMasses = 'data/processed/masses' + extraPath[i]
        
       
        # Create new directories if needed
        if not os.path.exists(procPathMasses):
            os.makedirs(procPathMasses)
        
    
        for filename in glob.glob(os.path.join(rawPath, '*.txt')):
            file = open (filename)
            # Calls functions from other scripts to encode the information in the '.txt' files
            attributes = encode_attributes(givenData=filename)
            lesion_type = file.readline()
            
            #Apenas se considera as imagens que apresentam um único tipo de margem
            if (len(attributes[1])==1 and attributes[1]!='0' and (attributes[0]=='1' or attributes[0]=='4' or attributes[0]=='6' or attributes[0]=='7' or attributes[0]=='8')):  
                if lesion_type.find("MASS") != -1:
                    if lesion_type.find("CALCIFICATION") == -1:
                        procPath = procPathMasses
                        countMasses += 1
                        count = countMasses

                    
                    
                # Opens the correspondent image and creates new images with its patches in the floder with processed data
                imgName = filename[0:len(filename)-6] + '.png'
                maskName = filename[0:len(filename)-3] + 'npy'
                patch_arr = get_mask_patch(imgName, maskName, IMG_SIZE)
                patch = Image.fromarray(patch_arr)
                patch.save (procPath + '\\' + str(count) + '.png')
    
    
                # Adds information about the current file to the dictionaries
                if lesion_type.find("MASS") != -1:
                    if lesion_type.find("CALCIFICATION") == -1:
                        yM["imgName"].append(str(count) + '.png')   
                        attrM["imgName"].append (str(count) + '.png')
                        if attributes[0]=='4':  #irregular que era 4 passa a 0
                            attributes[0]='0'
                        elif attributes[0]=='7':  #oval que era 7 passa a 1
                            attributes[0]='1'
                        elif attributes[0]=='6':  #lobulated que era 6 passa a 2
                            attributes[0]='2'
                        elif attributes[0]=='1':  #distortion que era 1 passa a 3
                            attributes[0]='3'
                        else:  #round que era 8 passa a 4
                            attributes[0]='4'
                            
                        #Como não se considera N/a que era o 0, todas as classes andam 1 valor para trás
                        attributes[1]=str(int(attributes[1])-1)

                        attrM["attr1"].append(attributes[0])
                        attrM["attr2"].append(attributes[1])

    
    
            # Uses Pandas library to convert the dictionaries information to .csv files
            yM_df = pd.DataFrame(yM)
            yM_df.to_csv(procPathMasses + '\\' + 'y.csv', header=True, index=False)
            attrM_df = pd.DataFrame(attrM)
            attrM_df.to_csv(procPathMasses + '\\' + 'attr.csv', header=True, index=False)
            


    print("Finished.")

if __name__ == '__main__':
    create_dataset()
