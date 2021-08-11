# -*- coding:utf-8 -*-
"""
 @Time: 2021/8/11 下午2:36
 @Author: LiuHe
 @File: BiLSTM_attention.py
 @Describe:
"""
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


class AttentionLayer(Layer):
    def __init__(self, attention_size=None,):
        self.attention_size = attention_size
        super(AttentionLayer, self).__init__()

    def get_config(self):
        config = super().get_config()
        config['attention_size'] = self.attention_size
        return config

    def build(self, input_shape):
        self.time_steps = input_shape[1]
        hidden_size = input_shape[2]
        if self.attention_size is None:
            self.attention_size = hidden_size

        self.W = self.add_weight(name='att_weight', shape=(hidden_size,
                                                           self.attention_size), initializer='uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(self.attention_size,
                                                         ),
                                 initializer='uniform', trainable=True)
        self.V = self.add_weight(name='att_var', shape=(self.attention_size,
                                                        ),
                                 initializer='uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        self.V = K.reshape(self.V, (-1, 1))
        H = K.tanh(K.dot(inputs, self.W)+self.b)
        score = K.softmax(K.dot(H, self.V), axis=1)
        outputs = K.sum(score * inputs, axis=1)
        return outputs


class BiLSTM(Model):
    def __init__(self, vocab_size, embedding_dim, hidden_units, class_num):
        super(BiLSTM, self).__init__()
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim)
        self.bi_lstm = keras.layers.Bidirectional(
            keras.layers.LSTM(hidden_units, return_sequences=True)
        )
        self.attention = AttentionLayer()
        self.outputs = keras.layers.Dense(class_num, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        embedded = self.embedding(inputs)
        bi_lstm_output = self.bi_lstm(embedded)
        att_output = self.attention(bi_lstm_output)
        output = self.outputs(att_output)

        return output


if __name__ == '__main__':
    sentences = ['i love you', 'he loves me', 'she likes baseball',
                 'i hate you', 'sorry for that', 'this is awful']
    labels = [1, 1, 1, 0, 0, 0]

    vocab_size = 20
    embedding_dim = 10
    max_length = 10
    hidden_units = 10
    class_nums = 2

    tokinzer = Tokenizer(num_words=vocab_size)
    tokinzer.fit_on_texts(sentences)
    train_x = tokinzer.texts_to_sequences(sentences)
    train_x = pad_sequences(train_x, maxlen=max_length)

    train_y = keras.utils.to_categorical(labels)

    model = BiLSTM(vocab_size, embedding_dim, hidden_units, class_nums)
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    history = model.fit(train_x, train_y, epochs=30)


    def plot_graphs(history, string):
        plt.plot(history.history[string])
        plt.xlabel('epoches')
        plt.ylabel(string)
        plt.show()


    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')