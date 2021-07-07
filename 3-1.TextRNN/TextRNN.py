# -*- coding:utf-8 -*-
"""
 @Time: 2021/7/7 下午2:34
 @Author: LiuHe
 @File: TextRNN.py
 @Describe:
"""
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K


class MinimalRNNCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        return output, [output]


class TextRNN(Model):
    def __init__(self,
                 cell,
                 vocab_size,
                 embedding_dim,
                 max_len,
                 class_num=2):
        """
        initialization
        :param vocab_size:
        :param embedding_dim:
        :param max_len:
        :param class_num:
        """
        super(TextRNN, self).__init__()
        self.cell = cell
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.class_num = class_num

        self.embedding = keras.layers.Embedding(self.vocab_size,
                                                self.embedding_dim,
                                                input_length=self.max_len)
        self.rnn = keras.layers.RNN(self.cell)
        self.classifier = keras.layers.Dense(class_num, activation='sigmoid')

    def call(self, inputs):
        embedding = self.embedding(inputs)
        rnn = self.rnn(embedding)
        output = self.classifier(rnn)

        return output


if __name__ == '__main__':
    sentences = ['i love you', 'he loves me', 'she likes baseball',
                 'i hate you', 'sorry for that', 'this is awful']
    labels = [1, 1, 1, 0, 0, 0]

    vocab_size = 20
    embedding_dim = 10
    max_length = 10

    tokinzer = Tokenizer(num_words=vocab_size)
    tokinzer.fit_on_texts(sentences)
    train_x = tokinzer.texts_to_sequences(sentences)
    train_x = pad_sequences(train_x, maxlen=max_length)

    train_y = keras.utils.to_categorical(labels)

    cell = MinimalRNNCell(32)
    model = TextRNN(cell, vocab_size, embedding_dim, max_length)
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