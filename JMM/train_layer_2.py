#coding=utf-8
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

MY_RANDOM_STATE = 2019
torch.manual_seed(MY_RANDOM_STATE)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
FILE_MODEL_TMP = "model_tmp.pkl"
MODEL_DIR = 'model_seed' + str(MY_RANDOM_STATE)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
epoches = 20

path1 = r"C:\Users\Mmjiang\Desktop\ML_homework\Layer2-positive.txt"
maxlen = 0
num = 0
Acid = "ACDEFGHIKLMNPQRSTVWY"
zeros = np.zeros((20,))
total_arr_1 = []
with open(path1) as obj:
    for i in obj.readlines():
        if re.match('>',i)==None:
            num += 1
            i = list(i)
            #if len(i)>maxlen:maxlen=len(i)
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
            num += 1
            i = list(i)
            if len(i)>maxlen:maxlen=len(i)
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



total_arr_0_ind = []
path3 = r"C:\Users\Mmjiang\Desktop\ML_homework\Layer2-Ind-negative.txt"
with open(path3) as obj:
    for i in obj.readlines():
        if re.match('>',i)==None:
            num += 1
            i = list(i)
            #if len(i)>maxlen:maxlen=len(i)
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
            num += 1
            i = list(i)
            #if len(i)>maxlen:maxlen=len(i)
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

x_test = np.concatenate((total_arr_0_ind,total_arr_1_ind),axis=0)

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


def train_kfold():
    kf = KFold(n_splits=5, shuffle=True, random_state=MY_RANDOM_STATE)
    fold = 0
    for train_index, val_index in kf.split(x,np_y):#切分5次，每次产生一个验证集，一个训练集
        data_train, data_val = x[train_index], x[val_index]#得到对应的索引的训练集和测试集
        label_train, label_val = np_y[train_index], np_y[val_index]
        dataset = {"data_train": data_train, "label_train": label_train,  
            "data_val": data_val, "label_val": label_val }
        #print("#####################data_train#####################",len(dataset["label_val"]),dataset["label_val"])
        train_one_fold(dataset, fold)
        fold += 1

    

def train_one_fold(dataset,fold):
    train_set = DealDataset(dataset["data_train"], dataset["label_train"])#对训练集对应的训练标签
    val_set = DealDataset(dataset["data_val"], dataset["label_val"])#对验证集及对应的标签
    
    train_loader = DataLoader(dataset=train_set, batch_size=32,                              
                              shuffle=True, num_workers=4)
    
    val_loader = DataLoader(dataset=val_set, batch_size=32,                              
                              shuffle=False, num_workers=4)
    model = ConvNet().to(device='cuda')
    best_val_loss = 1000
    best_epoch = 0 
    
    file_model = "best_model_saved.pkl"
    
    train_loss = []
    val_loss = []
             
    for epoch in range(1,epoches):
        print("\n############### EPOCH : ", epoch, " ###############")
        epoch_loss_train_avg = train_one_epoch(model, train_loader, learning_rate)#内部分batch训练
        print("epoch_loss_train_avg: ", epoch_loss_train_avg)#1个epoch的平均损失
        
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, FILE_MODEL_TMP))#保存该model到临时文件
        
        file_model_tmp = os.path.join(MODEL_DIR, FILE_MODEL_TMP)#读取该临时文件中的model
        result_val = evaluate(file_model_tmp, val_loader)#用该model对验证集评估   
        #print(result_val)
        if best_val_loss > result_val['epoch_loss_avg']:#若在验证集上评估的结果更好，则
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, file_model))#保存至最终的文件
            best_val_loss = result_val['epoch_loss_avg']#更新最优损失值
            print("Save model, best_val_loss: ", best_val_loss)
            best_epoch = epoch 
        
        train_loss.append(epoch_loss_train_avg)
        val_loss.append(result_val['epoch_loss_avg'])
        
    plt.figure()
    #plt.subplot(1,2,1)
    plt.plot(range(1,epoches),train_loss,label="avg_train_loss")
    plt.plot(range(1,epoches),val_loss,'--',label="avg_val_loss")
    #plt.xlabel("epoch_index")
    plt.ylabel("Average loss ")
    plt.legend()
    plt.show()
    
            
    model.load_state_dict(torch.load(MODEL_DIR+"/best_model_saved.pkl"))
    model_fn = "hemo_{}_{}.pkl".format(fold, best_epoch)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, model_fn))
    
    #Save train_loss & val_loss
    with open(MODEL_DIR+"/logfile_loss_model_{}_{}.csv".format(fold, best_epoch), mode = 'w') as lf_loss:
        lf_loss = csv.writer(lf_loss, delimiter=',')
        lf_loss.writerow(['epoch', 'train loss', 'validation loss'])
        for i in range(np.size(train_loss)):
            lf_loss.writerow([i, train_loss[i], val_loss[i]])
    
    print("\n####### EVALUATE ON VALIDATION ")
    print("best_epoch: ", best_epoch)
    
    print("\n ==> VALIDATION RESULT: ") 
    file_model =  os.path.join(MODEL_DIR, model_fn) #验证集上最好的model保存至该变量
    result_val = evaluate(file_model, val_loader)
    
         
         
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
    return epoch_loss_train_avg
    
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
        print(labels.shape)
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
    print("arr_labels_hyp: ", len(arr_labels_hyp))
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
        
tensor_x = torch.from_numpy(x).to(device).float()
#print(tensor_x.shape)
np_y_1 = np.ones((total_arr_1.shape[0],1))
np_y_0 = np.zeros((total_arr_0.shape[0],1))
np_y = np.concatenate((np_y_0,np_y_1),axis=0)
    
tensor_y = torch.from_numpy(np_y).to(device).float().view(-1,1)
#tensor_y = torch.LongTensor(tensor_y)
#print(tensor_y.shape)
    
#测试集
    
tensor_x_test = torch.from_numpy(x_test).to(device).float()
np_y_1_test = np.ones((total_arr_1_ind.shape[0],))
np_y_0_test = np.zeros((total_arr_0_ind.shape[0],))
np_y_test = np.concatenate((np_y_0_test,np_y_1_test),axis=0)
#print(np_y_test.shape)#要转化为2dim
tensor_y_test = torch.from_numpy(np_y_test).to(device).long().view(-1,1)


#model = ConvNet(2).to(device='cuda')
best_test_acc = 0


learning_rate = 1e-3


if __name__== "__main__":
    traininglog_fn = MODEL_DIR + "/training_log.csv"
    
    train_kfold() 
