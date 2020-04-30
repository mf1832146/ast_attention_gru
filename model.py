from keras.models import Model
from keras.layers import Input, Maximum, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten, Bidirectional, CuDNNGRU, RepeatVector, Permute, TimeDistributed, dot
from keras.backend import tile, repeat, repeat_elements
from keras.optimizers import RMSprop, Adamax
import keras
import keras.utils
import tensorflow as tf


class AstAttentionGRUModel:
    def __init__(self, config):
        config['code_len'] = 50
        config['sbt_len'] = 100
        config['com_len'] = 30

        self.config = config

        self.code_len = config['code_len']
        self.sbt_len = config['sbt_len']
        self.com_len = config['com_len']

        self.code_vocab_size = config['code_vocab_size']
        self.sbt_vocab_size = config['sbt_vocab_size']
        self.com_vocab_size = config['com_vocab_size']

        self.emb_dim = 100
        self.gru_dim = 256

    def create_model(self):

        code_input = Input(shape=(self.code_len,))
        com_input = Input(shape=(self.com_len,))
        sbt_input = Input(shape=(self.sbt_len,))

        code_emb = Embedding(output_dim=self.emb_dim, input_dim=self.code_vocab_size, mask_zero=False)(code_input)
        sbt_emb = Embedding(output_dim=self.emb_dim, input_dim=self.sbt_vocab_size, mask_zero=False)(sbt_input)

        sbt_enc = CuDNNGRU(self.gru_dim, return_state=True, return_sequences=True)
        sbt_out, state_sbt = sbt_enc(sbt_emb)

        code_enc = CuDNNGRU(self.gru_dim, return_state=True, return_sequences=True)
        code_out, state_code = code_enc(code_emb, initial_state=state_sbt)

        com_emb = Embedding(output_dim=self.emb_dim, input_dim=self.com_vocab_size, mask_zero=False)(com_input)
        decoder = CuDNNGRU(self.gru_dim, return_sequences=True)
        decode_output = decoder(com_emb, initial_state=state_code)

        attn = dot([decode_output, code_out], axes=[2, 2])
        attn = Activation('softmax')(attn)
        context = dot([attn, code_out], axes=[2, 1])

        ast_attn = dot([decode_output, sbt_out], axes=[2, 2])
        ast_attn = Activation('softmax')(ast_attn)
        ast_context = dot([ast_attn, sbt_out], axes=[2, 1])

        context = concatenate([context, decode_output, ast_context])

        out = TimeDistributed(Dense(self.gru_dim, activation="relu"))(context)

        out = Flatten()(out)
        out = Dense(self.com_vocab_size, activation="softmax")(out)

        model = Model(inputs=[code_input, com_input, sbt_input], outputs=out)

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return self.config, model


def create_model(config):
    mdl = AstAttentionGRUModel(config)
    return mdl.create_model()
