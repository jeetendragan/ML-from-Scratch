import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
#pd.set_option('display.width', 200)
import random
import threading
import time



class Node:
    def __init__(self, geni, data_mask):
        self.feature = None
        self.value = None
        self.geni = geni
        self.data_mask = data_mask
        self.left = None # this will store the left subtree root - Will be an instance of Node
        self.right = None # this will store the right subtree root - Will be an instance of the Node
        self.node_class = None # this will be non-None for leaf nodes only

    def bfs(self):
        conditions = []
        queue = []
        queue.append(self)
        while len(queue) > 0:
            node = queue.pop()
            if node.node_class != None:
                #conditions.append("Leaf node class: "+str(node.node_class))
                pass
            else:
                st = "Feature: " + node.feature
                st = st + ", Value: "+str(node.value)
                st = st + ", geni: "+str(node.geni)
                conditions.append(st)
                queue.append(node.left)
                queue.append(node.right)
        return conditions

    def predict_one(self, data_point):
        if self.node_class != None:
            return self.node_class
        if data_point[self.feature] <= self.value:
            return self.left.predict_one(data_point)
        else:
            return self.right.predict_one(data_point)

    def predict(self, data_frame):
        predictions = []
        for index, row in data_frame.iterrows():
            pred = self.predict_one(row)
            predictions.append(pred)
        return predictions


def get_majority_class(data: pd.DataFrame, class_column_name: str):
    return data[class_column_name].value_counts().index[0]

def calculate_geni(data, predictor_name):
    cnts = data[predictor_name].value_counts()
    total = data.shape[0]
    summation = 0
    for index in cnts.index:
        p_c = cnts[index]/total
        summation = summation + p_c**2
        
    geni = 1 - summation
    return geni

def get_random_features(features, m):
    rand_indices = random.sample(range(len(features)), m)
    return features[rand_indices]

def calculate_split_gain(node: Node, data: pd.DataFrame, predictor_name, feature: str, value: float):
    """ This method calculates the split gain at a node given the feature and value.
    The data has to be the complete data frame. 
    The data specific to this node is extracted by a set of indices stored in the node object.
    """
    this_node_data = data.iloc[node.data_mask]
    this_node_geni = node.geni
    left_node_data = this_node_data[this_node_data[feature] <= value]
    right_node_data = this_node_data[~(this_node_data[feature] <= value)]
    left_geni = calculate_geni(left_node_data, predictor_name)
    right_geni = calculate_geni(right_node_data, predictor_name)
    tot_cnt = this_node_data.shape[0]
    left_cnt = left_node_data.shape[0]
    right_cnt = right_node_data.shape[0]
    normalized_geni = left_geni * (left_cnt/tot_cnt) + right_geni * (right_cnt/tot_cnt)
    info_gain = this_node_geni - normalized_geni
    return {'info_gain': info_gain, 'left_geni': left_geni, 'right_geni': right_geni, 'left_mask': left_node_data.index, 'right_mask': right_node_data.index}

def calculate_best_split(data: pd.DataFrame, node, predictor_name, m):
    """
    This method calculates the best split possible at a node by randomly selecting m features.
    The data is the complete data frame and not specific to a node
    For each randomly generated feature, we get all the possible values for the current node data. 
    And then find the best value for each feature and the best feature-value overall.
    """
    feature_set = get_random_features(data.columns[0:len(data.columns)-1], m)
    best_info_gain = 0
    node_true_mask = left_geni = right_geni = best_feature = best_value = None
    this_node_data = data.iloc[node.data_mask]
    for feature in feature_set:
        feature_values = np.linspace(this_node_data[feature].min(), this_node_data[feature].max(), 4)
        #feature_values = sorted(this_node_data[feature].unique())
        for value in feature_values:
            result = calculate_split_gain(node, data, predictor_name, feature, value)
            if result['info_gain'] > best_info_gain:
                best_info_gain = result['info_gain']
                left_mask = result['left_mask']
                right_mask = result['right_mask']
                left_geni = result['left_geni']
                right_geni = result['right_geni']
                best_feature = feature
                best_value = value
    if best_info_gain == 0:
        return None
    else:
        return {'info_gain': best_info_gain, 'best_feature': best_feature, 'best_value': best_value,'left_geni': left_geni, 'right_geni': right_geni, 'left_mask': left_mask, 'right_mask': right_mask}

def build_node(node: Node, data: pd.DataFrame, predictor_name: str, m: int, max_size: int, max_depth: int , current_depth: int):
    """ This method adds a new node by selecting the best split possible(i.e. there is information gain)
    Note that the data is the main data frame with all the rows
    node contains a set of data indices specific it
    Only the leaf nodes will have the node_class not None.
    """
    if len(node.data_mask) < max_size or current_depth == max_depth:
        # if the node has less then max_size nodes, we don't split further
        node.node_class = get_majority_class(data.iloc[node.data_mask], predictor_name)
        return
    
    best_split = calculate_best_split(data, node, predictor_name, m)
    if best_split == None:
        # no better split is possible. Make this a leaf node.
        node.node_class = get_majority_class(data.iloc[node.data_mask], predictor_name)
        return
    
    #print("Best split - " + str(best_split['best_feature']) + ", " + str(best_split['best_value']))
    node.feature = best_split['best_feature']
    node.value = best_split['best_value']
    node.left = Node(best_split['left_geni'], best_split['left_mask'])
    node.right = Node(best_split['right_geni'], best_split['right_mask'])
    
    # see if it is possible to split the node on the left
    build_node(node.left, data, predictor_name, m, max_size, max_depth, current_depth + 1)
    
    # see if it is possible to split the node on the right
    build_node(node.right, data, predictor_name, m, max_size, max_depth, current_depth + 1)

def bootstrap(data: pd.DataFrame):
    n = data.shape[0]
    indices = [random.randint(0, n-1) for p in range(0, n)]
    bootstrap_sample = data.iloc[indices]
    return bootstrap_sample.reset_index(drop=True)

class RandomForest:
    def __init__(self, n, m, max_size, max_depth):
        self.trees = list()
        self.n = n
        self.m = m
        self.max_size = max_size
        self.max_depth = max_depth
    
    def build_random_forest(self, main_data: pd.DataFrame, predictor_name):
        threads = []
        for i in range(0, self.n):
            bootstrap_sample = bootstrap(main_data)
            print("Building tree: "+str(i+1)+".....")
            geni = calculate_geni(bootstrap_sample, predictor_name)
            node = Node(geni, range(0, bootstrap_sample.shape[0]))
            self.trees.append(node)
            #build_node(node, bootstrap_sample, predictor_name, self.m, self.max_size, self.max_depth, 0)
            x = threading.Thread(target=build_node, args=(node, bootstrap_sample, predictor_name, self.m, self.max_size, self.max_depth, 0,))
            x.start()
            threads.append(x)

        
        for thread in threads:
            thread.join()

        print("Forest built!")
        return self.trees
    
    def predict(self, data_frame):
        all_tree_pred = {}
        i = 1
        for tree in self.trees:
            tree_pred = tree.predict(data_frame)
            all_tree_pred[str(i)] = tree_pred
            i = i + 1
        df = pd.DataFrame(all_tree_pred)
        return df.mode(axis = 1)[0]

if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    train_size = int(data.shape[0]*0.7)
    indices = random.sample(range(0, data.shape[0]), train_size)
    data_train = data.iloc[indices]
    index_mask = data.index.isin(indices)
    data_test = data[~index_mask]
    geni = calculate_geni(data_train, '48')
    node = Node(geni, range(0, data_train.shape[0]))
    random_forest = RandomForest(12, 7, 20, 8)
    forest = random_forest.build_random_forest(data_train, '48')
    print("Predicting....")
    predictions = random_forest.predict(data_test)
    data_test_rst_index = data_test.reset_index(drop = True)
    accuracy = (predictions == data_test_rst_index['48']).sum()/data_test.shape[0]
    print("The accuracy obtained using random forest is: "+str(accuracy))
