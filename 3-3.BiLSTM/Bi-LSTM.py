# -*- coding:utf-8 -*-
"""
 @Time: 2021/7/7 下午4:27
 @Author: LiuHe
 @File: Bi-LSTM.py
 @Describe:
"""
from tensorflow import keras
from tensorflow.keras import Model
import numpy as np
from random import random
import matplotlib.pyplot as plt


class BiLSTM(Model):
    def __init__(self,
                 n_time_step):
        super(BiLSTM, self).__init__()
        self.n_time_step = n_time_step
        self.lstm = keras.layers.LSTM(20, input_shape=(self.n_time_step, 1),
                                      return_sequences=True)
        self.predict =keras.layers.TimeDistributed(
            keras.layers.Dense(1, activation='sigmoid')
        )

    def call(self, inputs):
        lstm = self.lstm(inputs)
        predict = self.predict(lstm)

        return predict


def data_generator(n_timesteps):
    """
    create data
    :param n_timesteps:
    :return:
    """
    X = np.array([random() for _ in range(n_timesteps)])
    limit = n_timesteps / 4.0
    y = np.array([0 if x < limit else 1 for x in np.cumsum(X)])
    X = X.reshape(1, n_timesteps, 1)
    y = y.reshape(1, n_timesteps, 1)
    return X, y


if __name__ == '__main__':
    n_time_step = 10
    model = BiLSTM(n_time_step)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    for epoch in range(1000):
        X, y = data_generator(n_time_step)
        model.fit(X, y, epochs=1, batch_size=1, verbose=2)


