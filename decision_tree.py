from collections import Counter
import numpy as np
import math
import sys
import csv



class commonTool():
    def getDataAndAttributes(self, filename):
        with open(filename) as f:
            temp = csv.reader(f, delimiter='\t')
            data = list(temp)
            data_without_head =np.array(data[1:].copy()).astype(object)
            attributes = np.array(data[0]).astype(object)
            return data_without_head, attributes
        
    def write_label_file(self, predict, labelFile):
        with open(labelFile, 'w') as f:
            for i in range(0, len(predict)-1):
                f.write(str(predict[i]) + '\n')
            f.write(str(predict[-1]))

    def write_metrics_file(self, train_error, test_error, target_metrics_file):
        with open(target_metrics_file, 'w') as f:
            f.write('error(train): ' + format(train_error, '.6f') + '\n')
            f.write('error(test): ' + format(test_error, '.6f'))

    def cal_entropy(self, labels):
        label_cnt = list(Counter(labels.tolist()).values())
        return sum([- i / len(labels) * np.log2(i / len(labels)) for i in label_cnt])

    def cal_mutual_information(self, labels, target_attribute_data_column):
        target_attribute_value = np.unique(target_attribute_data_column)
        mutual_information  = self.cal_entropy(labels)
        for value in target_attribute_value:
            mutual_information  -= self.cal_conditional_entropy(labels, target_attribute_data_column, value)
        return mutual_information

    def cal_conditional_entropy(self,labels,target_attribute_data_column, attribute_value):
        prob = np.count_nonzero(target_attribute_data_column == attribute_value) / len(target_attribute_data_column)
        entropy = self.cal_entropy(labels[target_attribute_data_column == attribute_value])
        return prob * entropy
    
    def cal_error_rate(self, predict, target):
        return np.count_nonzero(predict != target) / len(target)

class Node():
    def __init__(self, newCommonTool, train_data_without_label, train_data_label, depth, index, all_attr, max_depth, father = None):
        self.commonTool = newCommonTool
        self.train_data_without_label     = train_data_without_label
        self.train_data_label     = train_data_label
        self.label = self.get_majority_vote_label(train_data_label)
        self.depth = depth
        self.index   = None
        self.train_data_without_label_column  = index
        self.all_attr  = all_attr
        self.attr = None
        self.max_depth = max_depth
        self.father = father
        self.left_child   = None
        self.right_child  = None
        self.left_child_label = None
        self.right_child_label = None

    def get_majority_vote_label(self, Y):
        dict = {}
        for value in Y:
            if value not in dict:
                dict[value] = 0
            dict[value] += 1
        highest = sorted(dict.items(), key=lambda x: x[1], reverse=True)[0][1]
        for pair in dict.items():
            if(pair[1]== highest):
                return pair[0]

                      
    def choose_best_attribute(self):
        mutual = []
        for i in self.train_data_without_label_column:
            mutual.append(self.commonTool.cal_mutual_information(self.train_data_label, self.train_data_without_label[:, i]))
        max_mutual = max(mutual)
        return self.train_data_without_label_column[mutual.index(max_mutual)]

    def split(self):
        best_attribute_idx = self.choose_best_attribute()
        self.index = best_attribute_idx
        self.attr = self.all_attr[best_attribute_idx]
        values_set_best_attri = set(x for x in self.train_data_without_label[:,best_attribute_idx])
        values_array = np.zeros(len(values_set_best_attri),dtype='object')
        count = 0
        for value in values_set_best_attri:
            values_array[count] = value
            count = count + 1

        new_attribute_list = self.train_data_without_label_column.copy()
        new_attribute_list.remove(best_attribute_idx)
        for ele in values_array:
            is_element_present = self.train_data_label[self.train_data_without_label[:, best_attribute_idx] == ele].any()
            if not is_element_present:
                continue
            new_data_without_label = np.empty(shape=[0, len(self.train_data_without_label[0])],dtype='object')
            new_data_label = np.empty(0,dtype='object')

            for i in range(0, len(self.train_data_without_label)):
                row = self.train_data_without_label[i]
                if row[best_attribute_idx] == ele:
                    new_data_label = np.insert(new_data_label,len(new_data_label), self.train_data_label[i] ,axis = 0)
                    new_data_without_label = np.insert(new_data_without_label,len(new_data_without_label), row ,axis = 0)

            newNode = Node(self.commonTool, new_data_without_label, new_data_label, 
                                    self.depth + 1, new_attribute_list, self.all_attr, self.max_depth, self)
            if len(values_array) == 2:
                if ele == values_array[0]:
                    self.left_child = newNode
                    self.left_child_label = values_array[0]
                else:
                    self.right_child = newNode
                    self.right_child_label = values_array[1]
            else:
                self.left_child = newNode
                self.left_child_label = values_array[0]


    def train(self):   
        if self.train_data_without_label_column and len(np.unique(self.train_data_label)) > 1 and self.depth < self.max_depth:
            self.split()
            for node in [self.left_child, self.right_child]:
                if node:
                    node.train()
        else:
            return
    
    
    def predict(self, test):
        result = []
        for row in test:
            current = self
            while current.left_child or current.right_child:
                if row[current.index] == current.left_child_label:
                    current = current.left_child
                elif row[current.index] == current.right_child_label:
                    current = current.right_child
                else:
                    raise Exception("Invalid Input")
            result.append(current.label)
        return np.array(result)
   

def main():
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    max_depth = int(sys.argv[3])
    train_label_file = sys.argv[4]
    test_label_file = sys.argv[5]
    metrics_file = sys.argv[6]
    
    newCommonTool = commonTool()
    train_data, train_attr = newCommonTool.getDataAndAttributes(train_file)
    test_data, test_attr = newCommonTool.getDataAndAttributes(test_file)
    

    attribute_index_list = []
    len_attribute = train_data.shape[1] - 1
    for i in range(0,len_attribute):
        attribute_index_list.append(i)
    
    train_data_without_label = train_data[:,:-1]
    train_data_label = train_data[:, -1]
    root = Node(newCommonTool, train_data_without_label, train_data_label, 1, 
                attribute_index_list, 
                train_attr, int(max_depth) + 1)
    root.train()
    
    train_result = root.predict(train_data)
    test_result = root.predict(test_data)
    train_error = newCommonTool.cal_error_rate(train_result, train_data[:, -1])
    test_error  = newCommonTool.cal_error_rate(test_result, test_data[:, -1])
    newCommonTool.write_label_file(train_result,train_label_file)
    newCommonTool.write_label_file(test_result,test_label_file)
    newCommonTool.write_metrics_file(train_error,test_error, metrics_file)
    

if __name__ == '__main__':
    main()