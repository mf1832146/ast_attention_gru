import argparse
import json
import os
import pickle
import sys

import keras
import keras.backend as K
import numpy as np
import copy

from dataset import TestDataGen
from utils import load_pickle, load_json, gendescr_3inp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('modelfile', type=str, default=None)
    parser.add_argument('--num-procs', dest='numprocs', type=int, default='4')
    parser.add_argument('--gpu', dest='gpu', type=str, default='')
    parser.add_argument('--data_dir', default='../dataset')
    parser.add_argument('--outdir', dest='outdir', type=str, default='')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=200)
    parser.add_argument('--model-type', dest='modeltype', type=str, default=None)
    parser.add_argument('--outfile', dest='outfile', type=str, default=None)
    parser.add_argument('--zero-dats', dest='zerodats', action='store_true', default=False)
    parser.add_argument('--dtype', dest='dtype', type=str, default='float32')
    parser.add_argument('--tf-loglevel', dest='tf_loglevel', type=str, default='3')

    args = parser.parse_args()

    outdir = args.outdir
    outfile = args.outfile
    modelfile = args.modelfile

    if outfile is None:
        outfile = modelfile.split('/')[-1]

    K.set_floatx(args.dtype)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.tf_loglevel

    print('load data...')
    test_data = load_json(args.data_dir + '/test.json')

    code_i2w = load_pickle(args.data_dir + '/code_i2w.pkl')
    nl_i2w = load_pickle(args.data_dir + '/nl_i2w.pkl')
    ast_i2w = load_pickle(args.data_dir + '/ast_i2w.pkl')

    code_i2w[-1] = '<PAD>'
    nl_i2w[-1] = '<PAD>'
    ast_i2w[-1] = '<PAD>'

    code_w2i = {v: k + 1 for k, v in code_i2w.items()}
    nl_w2i = {v: k + 1 for k, v in nl_i2w.items()}
    ast_w2i = {v: k + 1 for k, v in ast_i2w.items()}

    code_i2w = {v: k for k, v in code_w2i.items()}
    nl_i2w = {v: k for k, v in nl_w2i.items()}
    ast_i2w = {v: k for k, v in ast_w2i.items()}

    test_ast_path = [x['ast_num'] for x in test_data]
    test_code = [x['code'] for x in test_data]
    test_nl = [x['nl'] for x in test_data]

    test_y = [[nl_w2i[t] if t in nl_w2i.keys() else nl_w2i["<UNK>"] for t in l] for l in test_nl]

    test_code_ids = [[code_w2i[t] if t in code_w2i.keys() else code_w2i["<UNK>"] for t in l] for l in test_code]

    print('loading model... ')
    model = keras.models.load_model(modelfile, custom_objects={})

    comstart = np.zeros(30)

    st = nl_w2i['<s>']
    comstart[0] = st

    outfn = outdir + "predict-{}.txt".format(outfile.split('.')[0])

    print("writing to file: " + outfn)

    config = dict()

    config['code_vocab_size'] = len(code_i2w) + 1
    config['sbt_vocab_size'] = len(ast_i2w) + 1
    config['com_vocab_size'] = len(nl_i2w) + 1

    test_gen = TestDataGen(test_code_ids, test_ast_path, test_nl, args.batch_size, ast_w2i,
                        path=args.data_dir + '/tree/test/', nl_dict_len=config['com_vocab_size'])

    iterator = test_gen.__iter__()

    comstart = np.zeros(30)
    st = nl_w2i['<s>']
    comstart[0] = st

    outputs = []
    for i in range(test_gen.__len__()):
        code_batch, sbt_batch, com_data, nodes_batch = next(iterator)

        com_start_list = [comstart for _ in range(len(code_batch))]
        com_start_batch = np.asarray(com_start_list)

        batch_results = gendescr_3inp(model, code_batch, com_start_batch, sbt_batch, nl_i2w, 30, args.batch_size)

        batch_output = [{'node_len': nodes_batch[i],
                         'predict:': batch_results[i],
                         'trues:': com_data[i]} for i in range(len(code_batch))]
        outputs.extend(batch_output)
        if i == 0:
            print(batch_output[0])

    with open(outfn, 'w') as f:
        json.dump(outputs, f)




