import argparse
import os
import time

from dataset import DataGen
from model import create_model
from util import load_json, read_pickle
from keras.callbacks import ModelCheckpoint, Callback
import atexit
import signal
import traceback
import sys
import pickle


class HistoryCallback(Callback):

    def setCatchExit(self, outdir, modeltype, timestart, mdlconfig):
        self.outdir = outdir
        self.modeltype = modeltype
        self.history = {}
        self.timestart = timestart
        self.mdlconfig = mdlconfig

        atexit.register(self.handle_exit)
        signal.signal(signal.SIGTERM, self.handle_exit)
        signal.signal(signal.SIGINT, self.handle_exit)

    def handle_exit(self, *args):
        if len(self.history.keys()) > 0:
            try:
                fn = self.outdir + '/histories/' + self.modeltype + '_hist_' + str(self.timestart) + '.pkl'
                histoutfd = open(fn, 'wb')
                pickle.dump(self.history, histoutfd)
                print('saved history to: ' + fn)

                fn = self.outdir + '/histories/' + self.modeltype + '_conf_' + str(self.timestart) + '.pkl'
                confoutfd = open(fn, 'wb')
                pickle.dump(self.mdlconfig, confoutfd)
                print('saved config to: ' + fn)
            except Exception as ex:
                print(ex)
                traceback.print_exc(file=sys.stdout)
        sys.exit()

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


if __name__ == '__main__':
    timestart = int(round(time.time()))
    parser = argparse.ArgumentParser(description='ast-attention gru')

    parser.add_argument('--epochs', dest='epochs', type=int, default=10)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=200)
    parser.add_argument('--data_dir', default='../dataset')

    args = parser.parse_args()

    """
    loading data
    """

    train_data = load_json(args.data_dir + '/train.json')
    valid_data = load_json(args.data_dir + '/valid.json')
    test_data = load_json(args.data_dir + '/test.json')

    code_i2w = read_pickle(args.data_dir + '/code_i2w.pkl')
    nl_i2w = read_pickle(args.data_dir + '/nl_i2w.pkl')
    ast_i2w = read_pickle(args.data_dir + '/ast_i2w.pkl')

    code_i2w[-1] = '<PAD>'
    nl_i2w[-1] = '<PAD>'
    ast_i2w[-1] = '<PAD>'

    code_w2i = {v: k+1 for k, v in code_i2w.items()}
    nl_w2i = {v: k+1 for k, v in nl_i2w.items()}
    ast_w2i = {v: k+1 for k, v in ast_i2w.items()}

    code_i2w = {v: k for k, v in code_w2i.items()}
    nl_i2w = {v: k for k, v in nl_w2i.items()}
    ast_i2w = {v: k for k, v in ast_w2i.items()}

    train_ast_path = [x['ast_num'] for x in train_data]
    train_code = [x['code'] for x in train_data]
    train_nl = [x['nl'] for x in train_data]

    valid_ast_path = [x['ast_num'] for x in valid_data]
    valid_code = [x['code'] for x in valid_data]
    valid_nl = [x['nl'] for x in valid_data]

    test_ast_path = [x['ast_num'] for x in test_data]
    test_code = [x['code'] for x in test_data]
    test_nl = [x['nl'] for x in test_data]

    train_y = [[nl_w2i[t] if t in nl_w2i.keys() else nl_w2i["<UNK>"] for t in l] for l in train_nl]
    valid_y = [[nl_w2i[t] if t in nl_w2i.keys() else nl_w2i["<UNK>"] for t in l] for l in valid_nl]
    test_y = [[nl_w2i[t] if t in nl_w2i.keys() else nl_w2i["<UNK>"] for t in l] for l in test_nl]

    train_code_ids = [[code_w2i[t] if t in code_w2i.keys() else code_w2i["<UNK>"] for t in l] for l in train_code]
    valid_code_ids = [[code_w2i[t] if t in code_w2i.keys() else code_w2i["<UNK>"] for t in l] for l in valid_code]
    test_code_ids = [[code_w2i[t] if t in code_w2i.keys() else code_w2i["<UNK>"] for t in l] for l in test_code]

    config = dict()

    config['code_vocab_size'] = len(code_i2w)
    config['sbt_vocab_size'] = len(ast_i2w)
    config['com_vocab_size'] = len(nl_i2w)

    config, model = create_model(config)

    train_gen = DataGen(train_code, train_ast_path, train_y, args.batch_size, ast_w2i, path=args.data_dir + '/tree/train/')
    valid_gen = DataGen(valid_code, valid_ast_path, valid_y, args.batch_size, ast_w2i, path=args.data_dir + '/tree/valid/')

    outdir = './out_dir'
    os.makedirs(outdir, exist_ok=True)
    checkpoint = ModelCheckpoint(outdir + '/models/' + 'E{epoch:02d}_' + str(timestart) + '.h5')
    savehist = HistoryCallback()

    callbacks = [checkpoint, savehist]
    steps = int(len(train_code) / args.batch_size) + 1
    valsteps = int(len(valid_code) / args.batch_size) + 1
    try:
        history = model.fit_generator(train_gen, steps_per_epoch=steps, epochs=args.epochs, verbose=1, max_queue_size=3, callbacks=callbacks, validation_data=valid_gen, validation_steps=valsteps)
    except Exception as ex:
        print(ex)
        traceback.print_exc(file=sys.stdout)
