# -*- coding:utf-8 -*-
"""
 @Time: 2021/7/15 上午10:38
 @Author: LiuHe
 @File: Seq2Seq.py
 @Describe: Change word
"""
from tensorflow import keras
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt


class Seq2Seq(Model):
    def __init__(self,
                 n_class,
                 hidden_dim):
        super(Seq2Seq, self).__init__()
        self.n_class = n_class
        self.hidden_dim = hidden_dim

        self.encoder = keras.layers.LSTM(self.hidden_dim, return_sequences=True)
        self.decoder = keras.layers.LSTM(self.hidden_dim, return_sequences=True)
        self.fc = keras.layers.Dense(self.n_class, activation='softmax')

    def call(self, inputs):
        encoder_outputs, state_h, state_c = self.encoder(inputs)
        encoder_state = [state_h, state_c]
        decoder_outputs, _, _ = self.decoder(encoder_state, initial_state=encoder_state)
        outputs = self.fc(decoder_outputs)

        return outputs


def get_data(seq_data, num_dic, n_class):
    """
    text sequences
    :param seq_data: text sequences
    :param num_dic:  dictionary
    :param n_class:  classes count
    :return:
    """
    input_data, output_data, target_data = [], [], []
    for seq in seq_data:
        for i in range(2):
            seq[i] = seq[i] + 'P'*(5-len(seq[i]))
        input = [num_dic[i] for i in seq[0]]
        output = [num_dic[i] for i in ('S'+seq[1])]
        target = [num_dic[i] for i in (seq[1]+'E')]
        input_data.append(np.eye(n_class)[input])
        output_data.append(np.eye(n_class)[output])
        target_data.append(target)

    return input_data, output_data, target_data


if __name__ == '__main__':
    # S: Symbol that shows starting of decoding input
    # E: Symbol that shows starting of decoding output
    # P: Symbol that will fill in blank sequence if current batch data size
    # is short than time steps

    char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz']
    num_dic = {n: i for i, n in enumerate(char_arr)}
    seq_data = [['man', 'woman'], ['black', 'white'], ['king', 'queen'],
                ['girl', 'boy'], ['up', 'down'], ['high', 'low']]

    n_class = len(num_dic)
    batch_size = len(seq_data)
    hidden_dim = 128

    input_data, output_data, target_data = get_data(seq_data, num_dic, n_class)
    print(np.array(input_data).shape)
    print(np.array(output_data).shape)
    print(np.array(target_data).shape)

    model = Seq2Seq(n_class, hidden_dim)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    history = model.fit(np.array([input_data, output_data]),
                        np.array(target_data), epochs=1000)

    def plot_graphs(history, string):
        plt.plot(history.history[string])
        plt.xlabel('epoches')
        plt.ylabel(string)
        plt.show()


    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')
