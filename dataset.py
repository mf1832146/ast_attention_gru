import json
import pickle
import random

import keras
import numpy as np

from utils import load_pickle


class DataGen(keras.utils.Sequence):
    def __init__(self, code_data, ast_data, com_data, batch_size, sbt_dic, path):
        self.code_data = code_data
        self.ast_data = ast_data
        self.com_data = com_data
        self.batch_size = batch_size
        self.sbt_dic = sbt_dic
        self.path = path
        self.allfids = range(len(self.code_data))

    def __len__(self):
        return int(np.ceil(len(self.code_data) / self.batch_size))

    def __getitem__(self, idx):
        start = (idx * self.batch_size)
        end = self.batch_size * (idx + 1)
        batchfids = self.allfids[start:end]

        code_data = [self.code_data[i] for i in batchfids]
        ast_data = [self.ast_data[i] for i in batchfids]
        com_data = [self.com_data[i] for i in batchfids]

        return self.gen(code_data, ast_data, com_data)

    def on_epoch_end(self):
        random.shuffle(self.allfids)

    def gen(self, code, ast_path, com):
        ast_tree = [load_pickle(self.path + n) for n in ast_path]
        sbt_tree = [sequencing(n) for n in ast_tree]

        sbt_pad = [pad([self.sbt_dic(t) for t in s], max_len=100) for s in sbt_tree]
        code_pad = [pad(s, max_len=50) for s in code]

        sbt_batch = []
        code_batch = []
        com_input_batch = []
        y_batch = []
        for j in len(com):
            sbt_j, code_j,com_input_j, y_j = generate_y(sbt_pad[j], code_pad[j], com[j], len(self.nl_dic))

            sbt_batch.extend(sbt_j)
            code_batch.extend(code_j)
            com_input_batch.extend(com_input_j)
            y_batch.extend(y_j)

        sbt_batch = np.asarray(sbt_batch)
        code_batch = np.asarray(code_batch)
        com_input_batch = np.asarray(com_input_batch)
        y_batch = np.asarray(y_batch)
        return [[code_batch, com_input_batch, sbt_batch], y_batch]


def generate_y(sbt, code, comment, com_vocab_size):
    sbt_batch = []
    code_batch = []
    com_input_batch = []
    y_batch = []
    for i in range(1, len(comment)):
        y = comment[i]
        com_input = comment[:i]

        y = keras.utils.to_categorical(y, num_classes=com_vocab_size)
        com_input = pad(com_input, max_len=30)

        y_batch.append(y)
        com_input_batch.append(com_input)

    sbt_batch.extend([sbt] * len(y_batch))
    code_batch.extend([code] * len(code_batch))

    return sbt_batch, code_batch, com_input_batch, y_batch


def sequencing(root):
    li = ["(", root.label]
    for child in root.children:
        li += sequencing(child)
    li += [")", root.label]
    return(li)


def pad(seq, max_len):
    if len(seq) < max_len:
        seq.extend([0] * max_len)
    return seq[:max_len]
