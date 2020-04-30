import json
import pickle


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def read_pickle(path):
    return pickle.load(open(path, "rb"))
