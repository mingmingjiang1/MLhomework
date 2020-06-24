#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import numpy as np
import pandas as pd
import csv

def write_to_svm(encodings, file):
    with open(file, 'a') as f:
        for line in encodings[1:]:
            line = line[1:]
            f.write('%s' % line[0])
            for i in range(1, len(line)):
                f.write('  %d:%s' % (i, line[i]))
            f.write('\n')


def save_file(encodings, format='svm', file='encodings.txt'):
    if encodings == 0:
        with open(file, 'a') as f:
            f.write('An error encountered.')
    else:
        if format == 'svm':
            write_to_svm(encodings, file)


def save_CV_result_binary(data, out, info=None):
    with open(out, 'w') as f:
        if info:
            f.write('%s\n' % info)
        for i in range(len(data)):
            f.write('# result for fold %d\n' %(i + 1))
            for j in range(len(data[i])):
                f.write('%d\t%s\n' % (data[i][j][0], data[i][j][2]))
    return None

def save_IND_result_binary(data, out, info=None):
    with open(out, 'w') as f:
        if info:
            f.write('%s\n' % info)
        for i in data:
            f.write('%d\t%s\n' % (i[0], i[2]))
    return None

def save_prediction_metrics_ind(m_dict, out):
    with open(out, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        head = []
        value = []
        for key in m_dict:
            head.append(key)
        csv_writer.writerow(head)

        for key in m_dict:
            value.append(m_dict[key])
        csv_writer.writerow(value)
    return None

def save_prediction_metrics_cv(m_list, out):
    with open(out, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        head = []
        head.append(' ')
        for key in m_list[0]:
            head.append(key)
        csv_writer.writerow(head)

        for i in range(len(m_list)):
            value = []
            value.append(i+1)
            for key in m_list[i]:
                value.append(m_list[i][key])
            csv_writer.writerow(value)
    return None

