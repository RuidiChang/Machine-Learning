#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:51:02 2023

@author: changruidi
"""

import csv
import sys

majority_vote = None

def majorityVote(data):
    zeros = 0
    ones = 0
    for i in range(1, len(data)):
        if (int(data[i][-1]) == 1):
            ones += 1
        else:
            zeros += 1
    global majority_vote
    
    if zeros > ones:
        majority_vote = 0 
    else:
        majority_vote = 1

def train(data):
    majority_vote = majorityVote(data)

def h(x):
    global majority_vote
    return majority_vote

def predict(data):
    result = []
    for i in range(1, len(data)):
        result.append(h(data[i]))
    return result

def evaluate(predict_array, original_dataList):
    incorrect = 0
    for i in range(0, len(predict_array)):
        if predict_array[i] != int(original_dataList[i+1][-1]):
            incorrect += 1
    return round(incorrect/len(predict_array),6)

def writeLabel(predict_array, target_label_file):
    with open(target_label_file, 'w') as f:
        for i in range(0, len(predict_array)-1):
            f.write(str(predict_array[i]) + '\n')
        f.write(str(predict_array[-1]))

def writeMetrics(train_error, test_error, target_metrics_file):
    with open(target_metrics_file, 'w') as f:
        f.write('error(train): ' + format(train_error, '.6f') + '\n')
        f.write('error(test): ' + format(test_error, '.6f'))

if __name__ == '__main__':
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    train_label_file = sys.argv[3]
    test_label_file = sys.argv[4]
    metrics_file = sys.argv[5]

    with open(train_file) as f:
        train_temp = csv.reader(f, delimiter='\t')
        train_data = list(train_temp)

    with open(test_file) as f:
        test_temp = csv.reader(f, delimiter='\t')
        test_data = list(test_temp)

    train(train_data)
    train_predict = predict(train_data)
    train_error = evaluate(train_predict, train_data)
    
    test_predict = predict(test_data)
    test_error = evaluate(test_predict, test_data)

    writeLabel(train_predict, train_label_file)
    writeLabel(test_predict, test_label_file)
    writeMetrics(train_error, test_error, metrics_file)
