# -*- encoding: utf-8 -*-
"""
@Time    : 2020/6/12
@Author  : lu
"""
import numpy as np
import math
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from dataset import readdata
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


def encoding(seq):
    account=[]
    account.append(seq.count('A')/len(seq))
    account.append(seq.count('R')/len(seq))
    account.append(seq.count('N')/len(seq))
    account.append(seq.count('D')/len(seq))
    account.append(seq.count('C')/len(seq))
    account.append(seq.count('Q')/len(seq))
    account.append(seq.count('E')/len(seq))
    account.append(seq.count('G')/len(seq))
    account.append(seq.count('H')/len(seq))
    account.append(seq.count('I')/len(seq))
    account.append(seq.count('L')/len(seq))
    account.append(seq.count('K')/len(seq))
    account.append(seq.count('M')/len(seq))
    account.append(seq.count('F')/len(seq))
    account.append(seq.count('P')/len(seq))
    account.append(seq.count('S')/len(seq))
    account.append(seq.count('T')/len(seq))
    account.append(seq.count('W')/len(seq))
    account.append(seq.count('Y')/len(seq))
    account.append(seq.count('V')/len(seq))
    return account

def load_data(file_name,data,label):
    with open(file_name,'r') as f:
        for num, value in enumerate(f):
            if num%2:
                data.append(value.strip('\n'))
            elif 'Negative' in value:
                label.append(0)

            else:
                label.append(1)
    return data,label

data_train=[]
label_train=[]
data_test=[]
label_test=[]

data_train,label_train=load_data('data/1/Layer1-negative.txt',data_train,label_train)
data_train,label_train=load_data('data/1/Layer1-positive.txt',data_train,label_train)
data_test,label_test=load_data('data/1/Layer1-Ind-negative.txt',data_test,label_test)
data_test,label_test=load_data('data/1/Layer1-Ind-positive.txt',data_test,label_test)

for i in range(len(data_train)):
    data_train[i]=encoding(data_train[i])
for i in range(len(data_test)):
    data_test[i]=encoding(data_test[i])

data_train=np.array(data_train)
label_train=np.array(label_train)
data_test=np.array(data_test)
label_test=np.array(label_test)

def calculate(arr_labels, arr_labels_hyp):
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

if __name__=='__main__':

    # train_data,test_data,train_label,test_label= readdata()

    X_train, Y_train= data_train,label_train
    X_test, Y_test = data_test,label_test
    #choose the k
    from sklearn.neighbors import KNeighborsClassifier

    neighbors = np.arange(1, 30)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))

    for i, k in enumerate(neighbors):
        # Setup a knn classifier with k neighbors
        knn = KNeighborsClassifier(n_neighbors=k)

        # Fit the model
        knn.fit(X_train, Y_train)

        # Compute accuracy on the training set
        train_accuracy[i] = knn.score(X_train, Y_train)

        # Compute accuracy on the test set
        test_accuracy[i] = knn.score(X_test, Y_test)

    plt.title('k-NN Varying number of neighbors')
    plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
    plt.plot(neighbors, train_accuracy, label='Training Accuracy')
    plt.legend()
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy')
    plt.show()


    # model1 = KNeighborsClassifier(n_neighbors=17)
    # model1.fit(X_train, Y_train)
    # score1 = model1.score(X_test, Y_test)

    knn = KNeighborsClassifier(n_neighbors=17)
    knn.fit(X_train, Y_train)

    Y_pred = knn.predict(X_test)
    confusion_matrix(Y_test, Y_pred)
    pd.crosstab(Y_test, Y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
    from sklearn.metrics import classification_report
    print(classification_report(Y_test, Y_pred))
    result1 = cross_val_score(knn, X_test, Y_test, cv=10)
    #
    print(result1)
    print(result1.mean())

    

    y_pred_proba = knn.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(Y_test, y_pred_proba)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Knn')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Knn(n_neighbors=17) ROC curve')
    plt.show()
    

    roc_score =roc_auc_score(Y_test, y_pred_proba)

