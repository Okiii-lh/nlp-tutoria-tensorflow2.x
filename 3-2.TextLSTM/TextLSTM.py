# -*- coding:utf-8 -*-
"""
 @Time: 2021/7/7 下午3:30
 @Author: LiuHe
 @File: TextLSTM.py
 @Describe:
"""
from tensorflow import keras
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt


class TextLSTM(Model):
    def __init__(self,
                 class_num):
        """
        initialization
        :param class_num:
        """
        super(TextLSTM, self).__init__()

        self.class_num = class_num

        self.lstm = keras.layers.LSTM(128)
        self.classifier = keras.layers.Dense(self.class_num,
                                             activation='sigmoid')

    def call(self, inputs):
        lstm = self.lstm(inputs)
        output = self.classifier(lstm)

        return output


def data_generator(sequence, word_dic):
    """
    create train data
    :param sequence: corpus
    :param word_dic: word dictionary
    :return: train_x, train_y
    """
    train_x = []
    train_y = []
    vocab_size = len(word_dict)
    for seq in sequence:
        x_word_indexs = [word_dict[n] for n in seq[:-1]]
        x = []
        for word_index in x_word_indexs:
            input = np.zeros(vocab_size)
            input[word_index] = 1
            x.append(input.tolist())
        y_word_index = word_dict[seq[-1]]
        y = np.zeros(vocab_size).tolist()
        y[y_word_index] = 1
        train_x.append(x)
        train_y.append(y)

    return train_x, train_y


if __name__ == '__main__':
    char_arr = [c for c in 'abcdefghijklmnopqrstuvwxyz']
    word_dict = {w: i for i, w in enumerate(char_arr)}
    number_dict = {i: w for i, w in enumerate(char_arr)}
    class_num = len(word_dict)

    seq_data = ['make', 'need', 'coal', 'word', 'love', 'hate', 'live',
                'home', 'hash', 'star']

    train_x, train_y = data_generator(seq_data, word_dict)
    print(np.array(train_x).shape)
    print(train_y)

    model = TextLSTM(class_num)
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    history = model.fit(train_x, train_y, epochs=1000)


    def plot_graphs(history, string):
        plt.plot(history.history[string])
        plt.xlabel('epoches')
        plt.ylabel(string)
        plt.show()


    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')