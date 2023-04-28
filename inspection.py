#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:51:02 2023

@author: changruidi
"""

import sys
import numpy as np
import math


class Inspection():
    
    def __init__(self,data):
        self.majority_vote = None
        self.result = []
        self.data = data
        lbset = set()
        for i in range(len(data)):
            lbset.add(data[i][-1])
        for i, v in enumerate(lbset):
            if (i == 0):
                self.label1 = v
            else:
                self.label2 = v
    
    def majorityVote(self,x):
        label1_count = 0
        label2_count = 0
        for i in range(len(self.data)):
            if (self.data[i][-1] == self.label1):
                label1_count += 1
            else:
                label2_count += 1
        if label1_count > label2_count:
            self.majority_vote = self.label1
        else:
            self.majority_vote = self.label2
        return self.majority_vote
    
    def predict(self):
        for i in range(len(self.data)):
            self.result.append(self.majorityVote(self.data[i]))
        return self.result
    
    def evaluate(self):
        self.result = self.predict()
        predict_array = self.result
        incorrect = 0
        for i in range(len(predict_array)):
            if predict_array[i] != int(self.data[i][-1]):
                incorrect += 1
        return round(incorrect/len(predict_array),6)
    
    def entropyCal(self):
        counter = {}
        for i in range(len(self.data)):
            label = self.data[i][-1]
            if label not in counter:
                counter[label] = 0
            counter[label] += 1
        entropy = 0
        for value in counter.values():
            entropy += -(value / len(self.data)) * math.log2(value / len(self.data))
        return entropy
        
def main():
    train_data = np.loadtxt(sys.argv[1], delimiter='\t', skiprows=1)
    ins = Inspection(train_data)
    output_file = sys.argv[2]
    entropy = ins.entropyCal()
    error = ins.evaluate()
    with open(output_file, 'w') as f:
        f.write("entropy: " + str(entropy))
        f.write('\n')
        f.write("error: " + str(error))

if __name__ == '__main__':
    main()