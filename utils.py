import json
import pickle


class Node:
    def __init__(self, label="", parent=None, children=[], num=0):
        self.label = label
        self.parent = parent
        self.children = children
        self.num = num


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def load_pickle(path):
    return pickle.load(open(path, "rb"))

