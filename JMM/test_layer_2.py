# -*- coding: GBK -*-
from sklearn import metrics
from sklearn.model_selection import KFold
import re
import os
import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn 
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
from sklearn import metrics
import math
import pickle
import time
import csv
import glob

learning_rate = 1e-3
MY_RANDOM_STATE = 2019
torch.manual_seed(MY_RANDOM_STATE)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
FILE_MODEL_TMP = "model_tmp.pkl"
MODEL_DIR = 'model_seed' + str(MY_RANDOM_STATE)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

epoches = 20
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
Acid = "ACDEFGHIKLMNPQRSTVWY"
#num=0

def train_one_epoch(model, train_loader,learning_rate):
    model.train()
    epoch_loss_train = 0.0
    nb_train = 0
    
   
    for i,data in enumerate(train_loader,0):
        x,y = data
        x = x.cuda()
        y = y.cuda().float()
        output = model(x)
        model.optimizer.zero_grad()
        loss = model.criterion(output,y)
        loss.backward()
        model.optimizer.step()
        epoch_loss_train += loss.item() * x.size(0)
        nb_train += x.size()[0]
    
    print("nb_train: ", nb_train)
    epoch_loss_train_avg = epoch_loss_train / nb_train
    #if (ep+1)%100 == 0:
    model.eval()
    with torch.no_grad():
        p_out = model(tensor_x)
        p_out = p_out.squeeze(1).cpu().numpy().tolist()
        #print(p_out)
        predict_labels = [int(prob > 0.5) for prob in p_out]
        #print(predict_labels)
        
        y_true = tensor_y.squeeze(1).cpu().numpy().tolist()
        arr_labels = [int(label) for label in y_true]
        #print(arr_labels)
        acc = np.sum(np.array(predict_labels) ==  np.array(arr_labels))/len(y_true)
        print('-- Acc of train : {}'.format(acc))
    return epoch_loss_train_avg


total_arr_0_ind = []
zeros = np.zeros((20,))
total_arr_1 = []
path1 = r"C:\Users\Mmjiang\Desktop\ML_homework\Layer2-positive.txt"
with open(path1) as obj:
    for i in obj.readlines():
        if re.match('>',i)==None:
            #num += 1
            i = list(i)
            for z in range(41-len(i)):
                i.append(0)
            arr = []
            for acid in Acid:
                acid_list = []
                for j in range(41):
                    if i[j]==acid: acid_list.append(1)
                    else: acid_list.append(0)
                arr.append(acid_list)
                
                
            total_arr_1.append(arr)
            
total_arr_1 = np.array(total_arr_1)

total_arr_0 = []
path2 = r"C:\Users\Mmjiang\Desktop\ML_homework\Layer2-negative.txt"
with open(path2) as obj:
    for i in obj.readlines():
        if re.match('>',i)==None:
            #num += 1
            i = list(i)
            for z in range(41-len(i)):
                i.append(0)  
            arr = []
            for acid in Acid:
                acid_list = []
                for j in range(41):
                    if i[j]==acid: acid_list.append(1)
                    else: acid_list.append(0)
                arr.append(acid_list)
                
                
            total_arr_0.append(arr)
            
total_arr_0 = np.array(total_arr_0)


path3 = r"C:\Users\Mmjiang\Desktop\ML_homework\Layer2-Ind-negative.txt"
with open(path3) as obj:
    for i in obj.readlines():
        if re.match('>',i)==None:
            #num += 1
            i = list(i)
            for z in range(41-len(i)):
                i.append(0) 
            arr = []
            for acid in Acid:
                acid_list = []
                for j in range(41):
                    if i[j]==acid: acid_list.append(1)
                    else: acid_list.append(0)
                arr.append(acid_list)
                
                
            total_arr_0_ind.append(arr)
            
total_arr_0_ind = np.array(total_arr_0_ind)


total_arr_1_ind = []
path4 = r"C:\Users\Mmjiang\Desktop\ML_homework\Layer2-Ind-positive.txt"
with open(path4) as obj:
    for i in obj.readlines():
        if re.match('>',i)==None:
            #num += 1
            i = list(i)
            for z in range(41-len(i)):
                i.append(0)
            arr = []
            for acid in Acid:
                acid_list = []
                for j in range(41):
                    if i[j]==acid: acid_list.append(1)
                    else: acid_list.append(0)
                arr.append(acid_list)
                
                
            total_arr_1_ind.append(arr)
            
total_arr_1_ind = np.array(total_arr_1_ind)




x = np.concatenate((total_arr_0,total_arr_1),axis=0)
tensor_x = torch.from_numpy(x).to(device).float()
np_y_1 = np.ones((total_arr_1.shape[0],1))
np_y_0 = np.zeros((total_arr_0.shape[0],1))
np_y = np.concatenate((np_y_0,np_y_1),axis=0)
trainset = {'data':x,"label":np_y}
tensor_y = torch.from_numpy(np_y).to(device).float().view(-1,1)

x_test = np.concatenate((total_arr_0_ind,total_arr_1_ind),axis=0)
tensor_x_test = torch.from_numpy(x_test).to(device).float()
np_y_1_test = np.ones((total_arr_1_ind.shape[0],1))
np_y_0_test = np.zeros((total_arr_0_ind.shape[0],1))
np_y_test = np.concatenate((np_y_0_test,np_y_1_test),axis=0)
#print(np_y_test.shape)#要转化为2dim
#tensor_y_test = torch.from_numpy(np_y_test).to(device).long().view(-1,1)
testset = {"data" : x_test, "label" : np_y_test}
tensor_y_test = torch.from_numpy(np_y_test).to(device).long().view(-1,1)


class DealDataset(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
        
    def __getitem__(self,index):
        x = self.x[index]
        y = self.y[index]
        input = torch.from_numpy(x).float()
        return input,y
        
    def __len__(self):
        return len(self.x)
        



class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        
        self.conv = torch.nn.Sequential()
        self.conv.add_module("conv_1",torch.nn.Conv1d(20,64,kernel_size=2))#
        self.conv.add_module("maxpool1",torch.nn.MaxPool1d(kernel_size=2))
        self.conv.add_module("dropout_1",torch.nn.Dropout())
        self.conv.add_module("relu_1",torch.nn.ReLU())
        self.conv.add_module("conv_2",torch.nn.Conv1d(64,128,kernel_size=2))
        self.conv.add_module("dropout_2",torch.nn.Dropout())
        self.conv.add_module("maxpool2",torch.nn.MaxPool1d(kernel_size=2))
        self.conv.add_module("relu_2",torch.nn.ReLU())
        
        self.fc = torch.nn.Sequential()
        self.fc.add_module("fc1",torch.nn.Linear(9*128,128))
        self.fc.add_module("relu_3",torch.nn.ReLU())
        self.fc.add_module("dropout_3",torch.nn.Dropout())
        self.fc.add_module("fc2",torch.nn.Linear(128,1))
        self.fc.add_module("sigmoid",torch.nn.Sigmoid())
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
    def forward(self,x):
        x = self.conv.forward(x)
        x = x.view(-1,9*128)
        return self.fc.forward(x)

def calculate_confusion_matrix(arr_labels, arr_labels_hyp):#计算混淆矩阵
    corrects = 0
    confusion_matrix = np.zeros((2, 2))

    for i in range(len(arr_labels)):
        confusion_matrix[arr_labels_hyp[i]][arr_labels[i]] += 1

        if arr_labels[i] == arr_labels_hyp[i]:
            corrects = corrects + 1

    acc = corrects * 1.0 / len(arr_labels)
    specificity = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])
    sensitivity = confusion_matrix[1][1] / (confusion_matrix[0][1] + confusion_matrix[1][1])
    tp = confusion_matrix[1][1]
    tn = confusion_matrix[0][0]
    fp = confusion_matrix[1][0]
    fn = confusion_matrix[0][1]
    if math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn)) == 0: mcc = 0
    else: mcc = (tp * tn - fp * fn ) / math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
    #print("mcc: ", mcc)
    return acc, confusion_matrix, sensitivity, specificity, mcc


def evaluate(file_model, loader):#评估
    #model.eval()
    model = ConvNet()
    #print("CNN Model: ", model)
    if torch.cuda.is_available(): model.cuda()
    
    model.load_state_dict(torch.load(file_model))
    model.eval()    
    
    epoch_loss = 0.0
    nb_samples = 0
    
    arr_labels = []
    arr_labels_hyp = []
    arr_prob = []
    
    for i, data in enumerate(loader, 0):
        # get the inputs
        inputs, labels = data
        #print("labels: ", labels)
        
        inputs_length = inputs.size()[0]
        nb_samples += inputs_length
        #print(labels.shape)
        arr_labels += labels.squeeze(1).data.cpu().numpy().tolist()

        inputs = inputs.float()
        labels = labels.float()
        
        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)
        
        outputs = model(inputs)
        loss = model.criterion(outputs, labels)
        
        epoch_loss = epoch_loss + loss.item() * inputs_length
        
        arr_prob += outputs.squeeze(1).data.cpu().numpy().tolist()
    
    print("nb_samples: ", nb_samples)
    #print("arr_prob: ", arr_prob)
    epoch_loss_avg = epoch_loss / nb_samples
    print("epoch loss avg: ", epoch_loss_avg)
    #arr_prob = np.array(arr_prob).max(axis=1)
    
   # print("arr_labels: ", arr_labels)
    
    auc = metrics.roc_auc_score(arr_labels, arr_prob)
    print("auc: ", auc)
    
    arr_labels_hyp = [int(prob > 0.5) for prob in arr_prob]
    #print("arr_prob: ", arr_prob)
    #print("arr_labels_hyp: ", len(arr_labels_hyp))
    arr_labels = [int(label) for label in arr_labels]
    
    acc, confusion_matrix, sensitivity, specificity, mcc = calculate_confusion_matrix(arr_labels, arr_labels_hyp)
    result = {'epoch_loss_avg': epoch_loss_avg, 
                'acc' : acc, 
                'confusion_matrix' : confusion_matrix,
                'sensitivity' : sensitivity,
                'specificity' : specificity,
                'mcc' : mcc,
                'auc' : auc,
                'arr_prob': arr_prob,
                'arr_labels': arr_labels,
                'arr_labels_hyp':arr_labels_hyp
                 }
    print("acc: ", acc)
    return result

def testing():
    #train_dataset = DealDataset(dataset["data"], dataset["label"])
    test_dataset = DealDataset(testset["data"], testset["label"])
    test_loader = DataLoader(dataset=test_dataset, batch_size=32,                              
                              shuffle=False, num_workers=4)
          
    with open(testresult_fn, mode='w') as outfile:
        outfile = csv.writer(outfile, delimiter=',')
        outfile.writerow(['model_fn', 'Accuracy score', 'AUC score', 'Sensitivity', 'Specificity','mcc'])
        
    list_model_fn = sorted(glob.glob(MODEL_DIR+"/hemo_*.pkl"))
    #print(list_model_fn)
    y_prob_mtx = []
    
    for model_fn in list_model_fn:
        print(model_fn)
        result = evaluate(model_fn, test_loader)
        print(result['arr_prob'])
        y_prob_mtx.append(result['arr_prob'])
        #break
        
        with open(testresult_fn, mode='a') as outfile:
            outfile = csv.writer(outfile, delimiter=',')
            outfile.writerow([model_fn, result['acc'], result['auc'], 
                result['sensitivity'], result['specificity'],result['mcc']])
    
    
    y_prob_mtx = np.array(y_prob_mtx)
    print("y_prob_mtx: ", y_prob_mtx.shape)
    print("y_prob_mtx: ", y_prob_mtx)
    
    y_prob_ensemble = [np.mean(y_prob_mtx[:,col]) for col in range(np.size(y_prob_mtx, 1))] 
    y_pred_ensemble = [np.float(each > 0.5) for each in y_prob_ensemble]
    
    y_true = testset["label"]
    auc_score_ensemble = metrics.roc_auc_score(y_true, y_prob_ensemble)
    accuracy_score_ensemble = metrics.accuracy_score(y_true, y_pred_ensemble)
    
    cm = metrics.confusion_matrix(y_true, y_pred_ensemble)
    specificity_ensemble = cm[0,0]/(cm[0,0] + cm[0,1])
    sensitivity_ensemble = cm[1,1]/(cm[1,1] + cm[1,0])

    print("Accuracy score (Testing Set) = ", accuracy_score_ensemble)
    print("ROC AUC score  (Testing Set) = ", auc_score_ensemble)
    print("Sensitivity    (Testing Set) = ", sensitivity_ensemble)
    print("Specificity    (Testing Set) = ", specificity_ensemble)
    
    with open(testresult_fn, mode='a') as outfile:
        outfile = csv.writer(outfile, delimiter=',')
        outfile.writerow(["ensemble", accuracy_score_ensemble, auc_score_ensemble, sensitivity_ensemble, specificity_ensemble])
        
        

    
if __name__== "__main__":
    testresult_fn = MODEL_DIR + "/test_result.csv"  
    testing()
