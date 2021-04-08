from collections import defaultdict
import pandas as pd
import numpy as np
import sys


def entropy(data):
    length = len(data)
    target = data.iloc[:, -1].value_counts()
    entropy = 0
    for i in target:
        p = i / length
        entropy -= p * np.log2(p)
    return entropy


def gain_ratio(data, attr):
    length = len(data)
    unique_val = data[attr].unique()
    gain = entropy(data)
    split_info = 0
    for i in unique_val:
        p = len(data[data[attr] == i]) / length
        split_info -= p * np.log2(p)
        gain -= p * entropy(data[data[attr] == i])

    return gain / split_info


class Node:
    def __init__(self, data, attr):
        if attr:
            self.attr = attr
            self.pointers = {k: 0 for k in data[attr].unique()}

    def __call__(self):
        print('node')


class Decision_Tree:
    def __init__(self):
        self.root = None

    def fit(self, data, root=True):
        if root:  # root case
            # stopping condition check
            if len(data.iloc[:, -1].value_counts()) == 1 or len(data.columns) == 1 or len(data) == 1:
                self.root = Node()
                max_val = 0
                # majority voting
                for k, v in data.iloc[:, -1].value_counts().iteritems():
                    if max_val < v:
                        max_val = v
                        majority = k
                self.root.pointers = {1: majority}
                return

            # select attribute that maximize gain ratio
            max_gain_ratio = 0
            for i in data.iloc[:, :-1].columns:
                gr = gain_ratio(data, i)
                if max_gain_ratio < gr:
                    max_gain_ratio = gr
                    max_col = i
            self.root = Node(data, max_col)

            # recursively works
            col = data.columns.drop(max_col)
            if len(col) > 1:
                for k in self.root.pointers.keys():
                    self.root.pointers[k] = self.fit(data[data[max_col] == k][col], False)
            else:
                self.root.leaf = True
                for key in self.root.pointers.keys():
                    max_val = 0
                    for k, v in data[data[self.root.attr] == key].iloc[:, -1].value_counts().iteritems():
                        if max_val < v:
                            max_val = v
                            majority = k
                    self.root.pointers[key] = majority
        else:
            # stopping condition check
            if len(data.iloc[:, -1].value_counts()) == 1 or len(data.columns) == 1 or len(data) == 1:
                max_val = 0
                # majority voting
                for k, v in data.iloc[:, -1].value_counts().iteritems():
                    if max_val < v:
                        max_val = v
                        majority = k
                return majority

            # select attribute that maximize gain ratio
            max_gain_ratio = 0
            for i in data.iloc[:, :-1].columns:
                gr = gain_ratio(data, i)
                if max_gain_ratio < gr:
                    max_gain_ratio = gr
                    max_col = i
            node = Node(data, max_col)

            # recursively works
            col = data.columns.drop(max_col)
            if len(col) > 1:
                for k in node.pointers.keys():
                    node.pointers[k] = self.fit(data[data[max_col] == k][col], False)
            else:
                # majority voting
                for key in node.pointers.keys():
                    max_val = 0
                    for k, v in data[data[node.attr] == key].iloc[:, -1].value_counts().iteritems():
                        if max_val < v:
                            max_val = v
                            majority = k
                    node.pointers[key] = majority

            return node

    def predict(self, data, indexes=None, current=None):
        if not current:  # root case
            indexes = set(data.index)
            all_indexes = indexes
            for k, v in self.root.pointers.items():
                new_indexes = indexes.intersection(set(data[data[self.root.attr] == k].index))
                all_indexes -= new_indexes
                if callable(v):
                    self.predict(data, new_indexes, v)
                else:
                    data.loc[new_indexes, data.columns[-1]] = v
            # class lable이 정해지지 못한 데이터가 있으면 leaf의 majority voting
            if all_indexes:
                self.result = defaultdict(lambda: 0)
                self.majority_voting(current)
                max_val = 0
                for k, v in self.result.items():
                    if max_val < v:
                        max_val = v
                        majority = k
                data.loc[all_indexes, data.columns[-1]] = majority
        else:
            all_indexes = indexes
            for k, v in current.pointers.items():
                new_indexes = indexes.intersection(set(data[data[current.attr] == k].index))
                all_indexes -= new_indexes
                if callable(v):
                    self.predict(data, new_indexes, v)
                else:
                    data.loc[new_indexes, data.columns[-1]] = v
            # class lable이 정해지지 못한 데이터가 있으면 아래 leaf의 majority voting
            if all_indexes:
                self.result = defaultdict(lambda: 0)
                self.majority_voting(current)
                max_val = 0
                for k, v in self.result.items():
                    if max_val < v:
                        max_val = v
                        majority = k
                data.loc[all_indexes, data.columns[-1]] = majority

    def majority_voting(self, current):
        for v in current.pointers.values():
            if callable(v):
                self.majority_voting(v)
            else:
                self.result[v] += 1

if __name__ == '__main__':
    train_df = pd.read_csv(sys.argv[1], sep = '\t')
    test_df = pd.read_csv(sys.argv[2], sep='\t')

    dt = Decision_Tree()
    dt.fit(train_df)
    test_df[train_df.columns[-1]] = None
    dt.predict(test_df)

    test_df.to_csv(sys.argv[3], index=False, sep='\t')
