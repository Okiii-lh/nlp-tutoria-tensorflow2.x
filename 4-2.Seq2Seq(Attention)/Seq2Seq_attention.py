# -*- coding:utf-8 -*-
"""
 @Time: 2021/7/23 下午2:26
 @Author: LiuHe
 @File: Seq2Seq_attention.py
 @Describe: Seq2Seq Attention
"""
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import activations
import matplotlib.pyplot as plt
import numpy as np


class Encoder(Model):
    def __init__(self, vocab_size, embedding_dim, hidden_units):
        super(Encoder, self).__init__()
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim,
                                                mask_zero=True)
        self.encoder_lstm = keras.layers.LSTM(hidden_units,
                                              return_sequences=True,
                                              return_state=True,
                                              name="encode_lstm")

    def call(self, inputs, training=None, mask=None):
        encoder_embedded = self.embedding(inputs)
        encoder_outputs, state_h, state_c = self.encoder_lstm(encoder_embedded)
        return encoder_outputs, state_h, state_c


class Decoder(Model):
    def __init__(self, vocab_size, embedding_dim, hidden_units):
        super(Decoder, self).__init__()
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim,
                                                mask_zero=True)
        self.decoder_lstm = keras.layers.LSTM(hidden_units,
                                              return_sequences=True,
                                              return_state=True,
                                              name="decode_lstm")
        self.attention = keras.layers.Attention()

    def call(self, inputs, en_outputs, states_inputs, raining=None, mask=None):
        decoder_embedded = self.embedding(inputs)
        decde_outputs, decde_state_h, decoder_state_c = self.decoder_lstm(
            decoder_embedded, initial_state=states_inputs)
        attention_output = self.attention([decde_outputs, en_outputs])

        return attention_output, decde_state_h, decoder_state_c


def Seq2Seq(max_len, embedding_dim, hidden_units, vocab_size):
    encoder_inputs = keras.layers.Input(shape=(max_len,), name="encode_input")
    decoder_inputs = keras.layers.Input(shape=(None,), name="decode_input")

    encoder = Encoder(vocab_size, embedding_dim, hidden_units)
    enc_outputs, enc_stats_h, enc_state_c = encoder(encoder_inputs)
    dec_states_inputs = [enc_stats_h, enc_state_c]

    decoder = Decoder(vocab_size, embedding_dim, hidden_units)
    attention_output, dec_state_h, dec_state_c = decoder(decoder_inputs,
                                                         enc_outputs, dec_states_inputs)
    dense_outputs = keras.layers.Dense(vocab_size, activation="softmax",
                                       name="decse")(attention_output)
    model = keras.models.Model([encoder_inputs, decoder_inputs],
                               dense_outputs)
    return model


max_len = 10
embedding_dim = 50
hidden_units = 128
vocab_size = 10000

model = Seq2Seq(max_len, embedding_dim, hidden_units, vocab_size)
print(model.summary())