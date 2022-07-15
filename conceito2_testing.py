# -*- coding: utf-8 -*-
"""
Created on Thu May 26 11:36:35 2022

@author:  Érica Gomes and Manuel Fortunato
"""

#Pytorch imports
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn as nn
#Other imports
import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# Confusion matrix and roc auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
#TSNE
from sklearn.manifold import TSNE
import matplotlib
# Import/Load bestModel 1
from conceito2_model import DatasetSimplifiedVersion, myModelPre

from itertools import cycle


if __name__ == "__main__":   
    bestModelS = myModelPre()
    bestModelS.load_state_dict(torch.load("myModelS_C2"))
    
    bestModelM = myModelPre()
    bestModelM.load_state_dict(torch.load("myModelM_C2"))    
        
    # check if CUDA is available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
        
    #Passing to GPU cuda
    bestModelS.to(device)
    bestModelM.to(device)
    
    
    transform_test = transforms.Compose([transforms.ToTensor(), 
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    transform_finaltest=transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # initialize batch_size
    batch_size = 16
    
    # building trainset, trainloader, testset and testloader
    csv_file = 'attr.csv' # 'att.csv'
    
    if device.type == 'cpu':
        initPath = r'C:\Users\efgom\OneDrive\Documentos\GitHub\breast-cancer-classification'
        num_workers = 0
    else:
        initPath = '/nas-ctm01-homes/efgomes'
        num_workers = 0

    root_dir_train = initPath + '/data/conceptl/conceito2/masses/train'
    root_dir_val = initPath + '/data/conceptl/conceito2/masses/val'
    root_dir_test = initPath + '/data/conceptl/conceito2/masses/test'
            
    testset = DatasetSimplifiedVersion(csv_file = csv_file, root_dir = root_dir_test, transform = transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = True, num_workers = num_workers)
       
    # defining the possible shapes and margins (used for confusion matrix)
    shape_classes = ('Irregular', 'Oval', 'Lobulated', 
                     'Architectural Distortion', 'Round')
    
    margin_classes = ('Circumscribed', 'Ill Defined',
                      'Microlobulated', 'Obscured', 'Spiculated')  
       
    
    # testing the trained model for shapes using finaltestset
    bestModelS.eval()
    
    correct_shape = 0
    total_shape = 0
    
    true_shape = np.array([])
    pred_shape = np.array([])
    testS_prev_before_out1=[]
    testS_truelabels=[]
    probS=[]

    
    with torch.no_grad():
        for data in testloader:
            images = data ['image'].to(device)
            shape = data ['shape'].to(device)
           
            x1= bestModelS.model(images)
            out1 = bestModelS.fc1(x1)
            _, predicted_shape = torch.max(out1.data, 1)
            
            total_shape += shape.size(0)
            correct_shape += (predicted_shape == shape).sum().item()
            
            # Building confusion Matrix arrays 
            true_shape = np.concatenate ((true_shape, shape.cpu().numpy()))
            pred_shape = np.concatenate ((pred_shape, predicted_shape.cpu().numpy()))
            
            testS_prev_before_out1.append((x1.cpu().numpy()))
            testS_truelabels.append((shape.cpu().numpy()))
            
            prob1=torch.softmax(out1.data,-1)
            probS.append(prob1.cpu().numpy())
        
        probS=np.concatenate(probS)
        testS_prev_before_out1 = np.concatenate(testS_prev_before_out1)
        testS_truelabels = np.concatenate(testS_truelabels)
        
    acc_shape_test = 100 * correct_shape // total_shape        
    
    # testing the trained model for margins using finaltestset
    bestModelM.eval()
    
    correct_margin = 0
    total_margin = 0
    
    true_margin = np.array([])
    pred_margin = np.array([])
    testM_prev_before_out1=[]
    testM_truelabels=[]
    probM=[]
    
    with torch.no_grad():
        for data in testloader:
            images = data ['image'].to(device)
            margin = data ['margin'].to(device)
            x2= bestModelM.model(images)
            out2 = bestModelM.fc2(x2)
            _, predicted_margin = torch.max(out2.data, 1)
            total_margin += margin.size(0)
            correct_margin += (predicted_margin == margin).sum().item()
            
            # Building confusion Matrix arrays 
            true_margin = np.concatenate ((true_margin, margin.cpu().numpy()))
            pred_margin = np.concatenate ((pred_margin, predicted_margin.cpu().numpy()))
            
            testM_prev_before_out1.append((x2.cpu().numpy()))
            testM_truelabels.append((margin.cpu().numpy()))
            
            prob2=torch.softmax(out2.data,-1)
            probM.append(prob2.cpu().numpy())
            
        probM=np.concatenate(probM)
        testM_prev_before_out1 = np.concatenate(testM_prev_before_out1)
        testM_truelabels = np.concatenate(testM_truelabels)

           
    acc_margin_test = 100 * correct_margin // total_margin      
    
    print('Best Model Accuracy in testset - shape : ',acc_shape_test, ' %')
    print('Best Model Accuracy in testset - margin : ', acc_margin_test, ' %')
    
    new_testS_truelabels=np.zeros((testS_truelabels.size,testS_truelabels.max()+1))
    new_testS_truelabels[np.arange(testS_truelabels.size),testS_truelabels]=1
    auc_valueS = roc_auc_score(new_testS_truelabels, probS,multi_class='ovr')
    print('AUC Shape: ', auc_valueS)
    
    new_testM_truelabels=np.zeros((testM_truelabels.size,testM_truelabels.max()+1))
    new_testM_truelabels[np.arange(testM_truelabels.size),testM_truelabels]=1
    auc_valueM = roc_auc_score(new_testM_truelabels, probM,multi_class='ovr')
    print('AUC Margin: ', auc_valueM)
    
            
    # vermelho, roxo, amarelo, verde, azul
    colors = cycle(["#ee4035", "#6d39b7", "#f4e659", "#7bc043", "#0392cf"])
    colors_TSNE=["#ee4035", "#6d39b7", "#f4e659", "#7bc043", "#0392cf"]
    cmap=matplotlib.colors.ListedColormap(colors_TSNE)
    
    fprS=dict()
    tprS=dict()
    roc_aucS=dict()
    for i in range(5):
        fprS[i], tprS[i], _ = roc_curve(new_testS_truelabels[:, i], probS[:, i])
        roc_aucS[i] = auc(fprS[i], tprS[i]) 

    lw=2
    for i, color in zip(range(5), colors):
        plt.plot(
            fprS[i],
            tprS[i],
            color=color,
            lw=lw,
            label="{0} (area = {1:0.2f})".format(shape_classes[i], roc_aucS[i]),)

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC curves for each class (shape)")
    plt.legend(loc="lower right")
    plt.show()
    
    # Save ROC-AUC margin figure into disk
    figure_name = 'C2_ROC-AUC_Shape' 
    figure_path = os.path.join(initPath + '/concept_learning/plots', f"{figure_name}")
    plt.savefig(fname = figure_path, bbox_inches = 'tight')
    plt.close()   
    
    fprM=dict()
    tprM=dict()
    roc_aucM=dict()
    for i in range(5):
        fprM[i], tprM[i], _ = roc_curve(new_testM_truelabels[:, i], probM[:, i])
        roc_aucM[i] = auc(fprM[i], tprM[i]) 
    
    
    lw=2
    for i, color in zip(range(5), colors):
        plt.plot(
            fprM[i],
            tprM[i],
            color=color,
            lw=lw,
            label="{0} (area = {1:0.2f})".format(margin_classes[i], roc_aucM[i]),)

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC curves for each class (margin)")
    plt.legend(loc="lower right")
    plt.show()
    
    # Save ROC-AUC margin figure into disk
    figure_name = 'C2_ROC-AUC_Margin' 
    figure_path = os.path.join(initPath + '/concept_learning/plots', f"{figure_name}")
    plt.savefig(fname = figure_path, bbox_inches = 'tight')
    plt.close()   

    
    # TSNE for shape
    testS_TSNE = TSNE().fit_transform(testS_prev_before_out1)
    tsne_shape= plt.scatter(*testS_TSNE.T, c=testS_truelabels, cmap=cmap, alpha=0.3)
    plt.legend(handles=tsne_shape.legend_elements()[0], labels=shape_classes)
    plt.title("TSNE for test set (shape)")
    plt.show()
    
    # Save TSNE shape figure into disk
    figure_name = 'C2_TSNE_Shape' 
    figure_path = os.path.join(initPath + '/concept_learning/plots', f"{figure_name}")
    plt.savefig(fname = figure_path, bbox_inches = 'tight')
    plt.close()
        
    #TSNE for margin
    testM_TSNE = TSNE().fit_transform(testM_prev_before_out1)
    tsne_margin= plt.scatter(*testM_TSNE.T, c=testM_truelabels, cmap=cmap, alpha=0.3)
    plt.legend(handles=tsne_margin.legend_elements()[0], labels=margin_classes)
    plt.title("TSNE for test set (margin)")
    plt.show()
    
    # Save TSNE margin figure into disk
    figure_name = 'C2_TSNE_Margin' 
    figure_path = os.path.join(initPath + '/concept_learning/plots', f"{figure_name}")
    plt.savefig(fname = figure_path, bbox_inches = 'tight')
    plt.close()
        
    # determine the confusion matrix for the data in the 'test' folder SHAPE
    confMatrix_shape = confusion_matrix(true_shape, pred_shape, normalize = 'true')
    disp_shape = ConfusionMatrixDisplay(confusion_matrix = confMatrix_shape, display_labels = shape_classes)
    disp_shape=disp_shape.plot(cmap="Blues", xticks_rotation=45)
    disp_shape.ax_.set_title('Confusion matrix for test set (shape)')
    
    # Save _Shape__ Confusion matrix figure into disk
    figure_name = 'C2_ConfMat_Shape' 
    figure_path = os.path.join(initPath + '/concept_learning/plots', f"{figure_name}")
    plt.savefig(fname = figure_path, bbox_inches = 'tight')
    plt.close()
    
    # determine the confusion matrix for the data in the 'test' folder MARGIN
    confMatrix_margin = confusion_matrix(true_margin, pred_margin, normalize = 'true')
    disp_margin = ConfusionMatrixDisplay(confusion_matrix = confMatrix_margin, display_labels = margin_classes)
    disp_margin=disp_margin.plot(cmap="Blues", xticks_rotation=45)
    disp_margin.ax_.set_title('Confusion matrix for test set (margin)')
    
    # Save Margin__ Confusion matrix figure into disk
    figure_name = 'C2_ConfMat_Margin' 
    figure_path = os.path.join(initPath + '/concept_learning/plots', f"{figure_name}")
    plt.savefig(fname = figure_path, bbox_inches = 'tight')
    plt.close()
