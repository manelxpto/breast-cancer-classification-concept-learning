# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 14:27:14 2022

@author: Maria e Manuel
"""


# Imports
import numpy as np
from PIL import Image



# Function: Get patch
def get_patch(img, coords_tuple, size):
    x, y = coords_tuple
    
    half_size = size // 2
    
    # Apply padding to image so that we can get patches from edges
    img = np.pad(img, half_size)
    
    # The patch is now obtained by looking at the coordinates (x:x+half_size, y:y+half_size), since we padded the image
    patch = img[x:x+2*half_size, y:y+2*half_size]

    return patch


# Function: Get center of a mask
def get_center(mask):
    
    for i in range (0, mask.shape[0]):
        if mask[i].sum() != 0:
            stLine = i
            break
    
    for j in range (mask.shape[0] - 1, -1, -1):
         if mask[j].sum() != 0:
            lastLine = j
            break
    
    for k in range (0, mask.shape[1]):
        if mask[:,k].sum() != 0:
            stCol = k
            break
    
    for l in range (mask.shape[1] - 1, -1, -1):
        if mask[:,l].sum() != 0:
            lastCol = l
            break 


    # The extra +1 is used to be coherent with get_patch when the side length is even
    col_center = int((lastCol + stCol + 1)/2)
    lin_center = int((lastLine + stLine + 1)/2)


    return (lin_center,col_center)



# Function: Get mask patch
def get_mask_patch(img_file: str, mask_file: str, size: int):
    im_frame = Image.open(img_file,'r')
    img = np.array(np.asarray(im_frame))
    mask = np.load(mask_file,'r')
    center = get_center(mask)
    x = get_patch(img, center, size)


    return x



# Test
if __name__ == "__main__":

    img1 = np.array ([[1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15],
                    [16, 17, 18, 19, 20],
                    [21, 22, 23, 24, 25]])

    patch_even = get_patch(img1, (2,2), 2)
    patch_odd = get_patch(img1, (2,2), 3)
    print (patch_even)
    print (patch_odd)


    img2 = np.array ([[0,0,0,0,0],
                    [0,1,1,1,0],
                    [0,1,1,1,0],
                    [0,0,0,0,0]])
    
    print(get_center(img2))
    # path = "C:/Users/david/OneDrive/Desktop/INESCTEC/ddsm/ddsm/test"
    # path1 = os.path.join(path,file+".png")
    # path2 = os.path.join(path,file+"_0.txt")
    # print(get_mask_patch(path1,path2,100))
