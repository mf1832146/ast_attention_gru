import json
import pickle
import numpy as np


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


def gendescr_3inp(model, code_data, com_out, sbt_data, com_i2w, com_len, batchsize):

    for i in range(1, com_len):
        results = model.predict([code_data, com_out, sbt_data], batch_size=batchsize)
        for c, s in enumerate(results):
            com_out[c][i] = np.argmax(s)

    final_data = []
    for i in range(len(com_out)):
        final_data[i] = seq2sent(com_out[i], com_i2w)

    return final_data


def seq2sent(seq, com_i2w):
    sent = []
    for i in seq:
        sent.append(com_i2w[i])

    return(' '.join(sent))

