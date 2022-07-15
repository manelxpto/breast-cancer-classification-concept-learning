# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 23:12:17 2022

@author: Manuel
"""
#Pytorch imports
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
#TSNE
from sklearn.manifold import TSNE
import matplotlib
from torch.utils.data import Dataset
#Other imports
from PIL import Image
import os
import tqdm
import numpy as np
import pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from skimage import io
import time

# this class creates a simplified dataset that only contains the information
# regarding shape and margin, ignoring the remaining attributes
class DatasetSimplifiedVersion (Dataset):
    def __init__ (self, csv_file, root_dir, transform = None):
        self.info = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__ (self):
        return len(self.info)
    
    def __getitem__ (self, index):
        # gets the important information about each image: its name and label
        img_name = os.path.join(self.root_dir, self.info.iloc[index, 0])
        image = io.imread(img_name)
        #label = self.info.iloc[index, 1]
        shape=self.info.iloc[index, 1]
        margin=self.info.iloc[index, 2]
        
        # we are working with grayscale images but resnet18 works with RGB 
        # images => there is the need to create a fake RGB image using ours as 
        # a starting point; to do so, we replicate the gray channel twice => 
        # we end up with 3 identical channels (this shouldn't affect model's 
        # efficiency)
        image = np.repeat(image[..., np.newaxis], 3, -1)
        image = Image.fromarray(image)
        sample = {'image': image, 'shape': shape, 'margin':margin}
        
        # applies the required transform to the image, mantaining the label
        if self.transform:
            transformedImg = self.transform(sample['image'])
            sample = {'image': transformedImg, 'shape': shape, 'margin':margin}

        return sample

class myModelPre(nn.Module):
    def __init__(self, num_shape=5, num_margin=5):
        super(myModelPre, self).__init__()
        self.model = models.resnet18(pretrained = True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.fc1 = nn.Linear(num_ftrs, num_shape)
        self.fc2 = nn.Linear(num_ftrs, num_margin)
    def forward(self, x):
        x = self.model(x)
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        return out1, out2

if __name__ == "__main__":
    # time counting
    since = time.time()
    
    # extra info for the .out file on WinSCP
    print('Conceito 2: "conceito2_model.py"\n')
    
    # check if CUDA is available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # the images aren't grayscale anymore when "transform" is called => we have to 
    # normalize 3 channels instead of 1
    
    
    transform_train = transforms.Compose([                                   
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomVerticalFlip(p=0.5),
                                    transforms.RandomRotation(degrees=(0,360)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
    
    transform_val = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    transform_test=transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # initialize batch_size
    batch_size = 16
    
    # building trainset, trainloader, testset and testloader
    csv_file = 'attr.csv' 
    
    if device.type == 'cpu':
        initPath = r'C:\Users\efgom\OneDrive\Documentos\GitHub\breast-cancer-classification'
        num_workers = 0
    else:
        initPath = '/nas-ctm01-homes/efgomes'
        num_workers = 0

    root_dir_train = initPath + '/data/conceptl/conceito2/masses/train'
    root_dir_val = initPath + '/data/conceptl/conceito2/masses/val'
    root_dir_test = initPath + '/data/conceptl/conceito2/masses/test'
    
    
    trainset = DatasetSimplifiedVersion (csv_file = csv_file, root_dir = root_dir_train, transform = transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, 
                                              shuffle = True, num_workers = num_workers)
    
    valset = DatasetSimplifiedVersion (csv_file = csv_file, root_dir = root_dir_val, transform = transform_val)
    valloader = torch.utils.data.DataLoader(valset, batch_size = batch_size, 
                                              shuffle = True, num_workers = num_workers)
    
    testset = DatasetSimplifiedVersion(csv_file = csv_file, root_dir = root_dir_test, transform = transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    
    # defining the possible shapes and margins (used for confusion matrix)
    shape_classes = ('Irregular', 'Oval', 'Lobulated', 
                     'Architectural Distortion', 'Round')
    
    margin_classes = ('Circumscribed', 'Ill Defined',
                      'Microlobulated', 'Obscured', 'Spiculated')  
    
    # model selection => pretrained resnet18 with a modification in the last layer 
    model = myModelPre()
    
    # defining the used method to calculate the loss and the used optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    
    #this variables save the maximum accuracy of the models during the training 
    #process
    accuracyMax_shape = 0
    accuracyMax_margin = 0
    epochMax_shape = -1
    epochMax_margin = -1
        
    # Variáveis para plot de gráficos e matrizes de confusão (por atualizar!)
    total_shape_train=0
    correct_shape_train=0
    total_margin_train=0
    correct_margin_train=0
    
    shapeAccTrain=[]
    marginAccTrain=[]
    shapeLossTrain=[]
    marginLossTrain=[]
    shapeAccVal=[]
    marginAccVal=[]
    valS_prev_before_out=[]
    valM_prev_before_out=[]
    valS_truelabels_best=[]
    valM_truelabels_best=[]
    
    # Se alfa=0, o modelo descarta a loss e treina só para o outro atributo
    alfa_s = 1
    alfa_m = 1
    
    #Passing to GPU cuda
    model.to(device)
    
    num_epochs=150
    
    # training the model using 'trainset'
    for epoch in tqdm.tqdm(range(num_epochs)): 
        running_loss1 = 0.0
        running_loss2 = 0.0
        # We put the model into training mode to make sure all the layers can be updated (e.g., dropout and batchnorm)
        model.train()
        
        for i, data in enumerate(trainloader, 0):
            inputs = data ['image'].to(device)
            shape = data ['shape'].to(device)
            margin = data ['margin'].to(device)
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            out1, out2 = model(inputs)
            loss1 = alfa_s * criterion(out1, shape)
            loss2= alfa_m * criterion(out2, margin)
            loss1.backward(retain_graph=True)
            loss2.backward(retain_graph=True)
            optimizer.step()
    
            #updating the loss value
            running_loss1 += loss1.cpu().detach()    
            running_loss2 += loss2.cpu().detach()  
            
            #accuracy for the train
            _, predicted_shape = torch.max(out1.data, 1)
            _, predicted_margin = torch.max(out2.data, 1)
            total_shape_train += shape.size(0)
            correct_shape_train += (predicted_shape == shape).sum().item()
            total_margin_train += margin.size(0)
            correct_margin_train += (predicted_margin == margin).sum().item()
    
        acc_shape_train=100*correct_shape_train//total_shape_train
        acc_margin_train=100*correct_margin_train//total_margin_train
    
        shapeAccTrain.append(acc_shape_train)
        marginAccTrain.append(acc_margin_train)
        shapeLossTrain.append((running_loss1/total_shape_train * batch_size).cpu().detach().numpy())
        marginLossTrain.append((running_loss2/total_margin_train * batch_size).cpu().detach().numpy())
                
        # testing the trained model using 'valset'
        val_prev_before_out=[]
        valS_truelabels=[]
        valM_truelabels=[]
        correct_shape = 0
        correct_margin = 0
        total_shape = 0
        total_margin = 0
        model.eval()
        
        with torch.no_grad():
            for data in valloader:
                images = data ['image'].to(device)
                shape = data ['shape'].to(device)
                margin = data ['margin'].to(device)
                
                # calculate outputs by running images through the network
                x= model.model(images)
                out1 = model.fc1(x)
                out2 = model.fc2(x)
                # the class with the highest energy is what we choose as prediction
                _, predicted_shape = torch.max(out1.data, 1)
                _, predicted_margin = torch.max(out2.data, 1)
            
                total_shape += shape.size(0)
                correct_shape += (predicted_shape == shape).sum().item()
                
                total_margin += margin.size(0)
                correct_margin += (predicted_margin == margin).sum().item()
                
                val_prev_before_out.append((x.cpu().numpy()))
                valS_truelabels.append((shape.cpu().numpy()))
                valM_truelabels.append((margin.cpu().numpy()))

            val_prev_before_out = np.concatenate(val_prev_before_out)
            valS_truelabels = np.concatenate(valS_truelabels)
            valM_truelabels = np.concatenate(valM_truelabels)
    
        acc_shape_val  = 100 * correct_shape // total_shape
        acc_margin_val = 100 * correct_margin // total_margin
        print(f'\nAccuracy of epoch (shape) {epoch + 1}: {acc_shape_val} % \n')
        print(f'\nAccuracy of epoch (margin) {epoch + 1}: {acc_margin_val} % \n')
    
        shapeAccVal.append(acc_shape_val)
        marginAccVal.append(acc_margin_val)
        
        # Salvar modelos ao longo do trai para ver evoluçao do TSNE
        if epoch==5:
            torch.save(model.state_dict(), 'Model_at_ep5_c2')            
        if epoch==25:
            torch.save(model.state_dict(), 'Model_at_ep25_c2')
        if epoch==50:
            torch.save(model.state_dict(), 'Model_at_ep50_c2')
        if epoch==100:
            torch.save(model.state_dict(), 'Model_at_ep100_c2')            
        if epoch==300:
            torch.save(model.state_dict(), 'Model_at_ep300_c2')    
       
        
        if acc_shape_val >= accuracyMax_shape:
            accuracyMax_shape = acc_shape_val
            torch.save(model.state_dict(), 'myModelS_C2')
            epochMax_shape = epoch
            
            valS_truelabels_best=valS_truelabels
            valS_prev_before_out=val_prev_before_out
            
            
        if acc_margin_val >= accuracyMax_margin:
            accuracyMax_margin = acc_margin_val
            torch.save(model.state_dict(), 'myModelM_C2')
            epochMax_margin = epoch
            
            valM_truelabels_best=valM_truelabels
            valM_prev_before_out=val_prev_before_out
            
        
    print(f'Finished Training -> Maximum Accuracy Reached - shape: {accuracyMax_shape} % at epoch {epochMax_shape + 1}')
    print(f'Finished Training -> Maximum Accuracy Reached - margin: {accuracyMax_margin} % at epoch {epochMax_margin + 1}')
        
    # Plot da Accuracy e da Loss
    epochs_x=list(range(1,num_epochs+1))
    fig, ax = plt.subplots(2,2,figsize=(10,8))
    ax[0][0].plot(np.array(epochs_x), np.array(shapeAccTrain), np.array(epochs_x), np.array(shapeAccVal))
    ax[0][0].set_title('Accuracy (shape)')
    ax[0][0].legend(['Accuracy for training set','Accuracy for validation set'])
    ax[0][1].plot(np.array(epochs_x), np.array(marginAccTrain), np.array(epochs_x), np.array(marginAccVal))
    ax[0][1].set_title('Accuracy (margin)')
    ax[0][1].legend(['Accuracy for training set','Accuracy for validation set'])
    ax[1][0].plot(np.array(epochs_x), np.array(shapeLossTrain))
    ax[1][0].set_title('Loss for training set (shape)')
    ax[1][1].plot(np.array(epochs_x), np.array(marginLossTrain))
    ax[1][1].set_title('Loss for training set (margin)')
    
    #Guardar plot
    figure_name = 'plots_concept2' 
    path = initPath +  '/concept_learning/plots'
    figure_path = os.path.join(path, f"{figure_name}")
    plt.savefig(fname = figure_path, bbox_inches = 'tight')  
    plt.close()
    
    # vermelho, roxo, amarelo, verde, azul
    colors = ["#ee4035", "#6d39b7", "#f4e659", "#7bc043", "#0392cf"]
    cmap=matplotlib.colors.ListedColormap(colors)
    
    # TSNE for shape
    valS_TSNE = TSNE().fit_transform(valS_prev_before_out)
    tsne_shape= plt.scatter(*valS_TSNE.T, c=valS_truelabels_best, cmap=cmap, alpha=0.3)
    plt.legend(handles=tsne_shape.legend_elements()[0], labels=shape_classes)
    plt.title("TSNE for validation set (shape)")
    plt.show()
    
    # Save TSNE shape figure into disk
    figure_name = 'C2_TSNEval_Shape' 
    figure_path = os.path.join(initPath + '/concept_learning/plots', f"{figure_name}")
    plt.savefig(fname = figure_path, bbox_inches = 'tight')
    plt.close()
        
    #TSNE for margin
    valM_TSNE = TSNE().fit_transform(valM_prev_before_out)
    tsne_margin= plt.scatter(*valM_TSNE.T, c=valM_truelabels_best, cmap=cmap, alpha=0.3)
    plt.legend(handles=tsne_margin.legend_elements()[0], labels=margin_classes)
    plt.title("TSNE for validation set (margin)")
    plt.show()
    
    # Save TSNE margin figure into disk
    figure_name = 'C2_TSNEval_Margin' 
    figure_path = os.path.join(initPath + '/concept_learning/plots', f"{figure_name}")
    plt.savefig(fname = figure_path, bbox_inches = 'tight')
    plt.close()
        
    
    # Criar/salvar o melhor modelo, fazendo load dos pesos
    bestModelS=myModelPre()  # criar o modelo
    bestModelS.load_state_dict(torch.load('myModelS_C2'))   # fazer load dos pesos
    
    bestModelM=myModelPre()  
    bestModelM.load_state_dict(torch.load('myModelM_C2'))     
    
    modelEp5=myModelPre()
    modelEp5.load_state_dict(torch.load('Model_at_ep5_c2'))
        
    modelEp25=myModelPre()
    modelEp25.load_state_dict(torch.load('Model_at_ep25_c2'))
    
    modelEp50=myModelPre()
    modelEp50.load_state_dict(torch.load('Model_at_ep50_c2'))
    
    modelEp100=myModelPre()
    modelEp100.load_state_dict(torch.load('Model_at_ep100_c2'))
    
    modelEp300=myModelPre()
    modelEp300.load_state_dict(torch.load('Model_at_ep300_c2'))
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))