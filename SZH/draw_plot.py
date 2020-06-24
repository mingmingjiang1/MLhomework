import os, platform, sys
import numpy as np
import pandas as pd
from itertools import cycle
from mpl_toolkits.mplot3d import Axes3D
from scipy import interp
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
pPath = os.path.split(os.path.realpath(__file__))[0]
father_path = os.path.abspath(
    os.path.dirname(pPath) + os.path.sep + ".") + r'\clusters' if platform.system() == 'Windows' else os.path.abspath(
    os.path.dirname(pPath) + os.path.sep + ".") + r'/clusters'
sys.path.append(father_path)


def plot_roc_cv(data, out, label_column=0, score_column=2):
    tprs = []
    aucs = []
    fprArray = []
    tprArray = []
    thresholdsArray = []
    mean_fpr = np.linspace(0, 1, 100)

    for i in range(len(data)):
        fpr, tpr, thresholds = roc_curve(data[i][:, label_column], data[i][:, score_column])
        fprArray.append(fpr)
        tprArray.append(tpr)
        thresholdsArray.append(thresholds)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'blueviolet', 'deeppink', 'cyan'])
    ## ROC plot for CV
    fig = plt.figure(0)
    for i, color in zip(range(len(fprArray)), colors):
        plt.plot(fprArray[i], tprArray[i], lw=1, alpha=0.7, color=color,
                 label='ROC fold %d (AUC = %0.2f)' % (i + 1, aucs[i]))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Random', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='blue',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.9)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(out)
    plt.close(0)
    return mean_auc

def plot_roc_ind(data, out, label_column=0, score_column=2):
    fprIndep, tprIndep, thresholdsIndep = roc_curve(data[:, label_column], data[:, score_column])
    ind_auc = auc(fprIndep, tprIndep)
    fig = plt.figure(0)
    plt.plot(fprIndep, tprIndep, lw=2, alpha=0.7, color='red',
             label='ROC curve (area = %0.2f)' % ind_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(out)
    plt.close(0)
    return ind_auc

def plot_prc_CV(data, out, label_column=0, score_column=2):
    precisions = []
    aucs = []
    recall_array = []
    precision_array = []
    mean_recall = np.linspace(0, 1, 100)

    for i in range(len(data)):
        precision, recall, _ = precision_recall_curve(data[i][:, label_column], data[i][:, score_column])
        recall_array.append(recall)
        precision_array.append(precision)
        precisions.append(interp(mean_recall, recall[::-1], precision[::-1])[::-1])
        roc_auc = auc(recall, precision)
        aucs.append(roc_auc)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'blueviolet', 'deeppink', 'cyan'])
    ## ROC plot for CV
    fig = plt.figure(0)
    for i, color in zip(range(len(recall_array)), colors):
        plt.plot(recall_array[i], precision_array[i], lw=1, alpha=0.7, color=color,
                 label='PRC fold %d (AUPRC = %0.2f)' % (i + 1, aucs[i]))
    mean_precision = np.mean(precisions, axis=0)
    mean_recall = mean_recall[::-1]
    mean_auc = auc(mean_recall, mean_precision)
    std_auc = np.std(aucs)

    plt.plot(mean_recall, mean_precision, color='blue',
             label=r'Mean PRC (AUPRC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.9)
    std_precision = np.std(precisions, axis=0)
    precision_upper = np.minimum(mean_precision + std_precision, 1)
    precision_lower = np.maximum(mean_precision - std_precision, 0)
    plt.fill_between(mean_recall, precision_lower, precision_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")
    plt.savefig(out)
    plt.close(0)
    return mean_auc

def plot_prc_ind(data, out, label_column=0, score_column=2):
    precision, recall, _ = precision_recall_curve(data[:, label_column], data[:, score_column])
    ind_auc = auc(recall, precision)
    fig = plt.figure(0)
    plt.plot(recall, precision, lw=2, alpha=0.7, color='red',
             label='PRC curve (area = %0.2f)' % ind_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")
    plt.savefig(out)
    plt.close(0)
    return ind_auc