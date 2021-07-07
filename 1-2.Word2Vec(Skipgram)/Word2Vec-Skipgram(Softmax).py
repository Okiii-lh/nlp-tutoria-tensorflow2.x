# -*- coding:utf-8 -*-
"""
 @Time: 2021/7/5 下午10:09
 @Author: LiuHe
 @File: Word2Vec-Skipgram(Softmax).py
 @Describe:
"""
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
import numpy as np
import matplotlib.pyplot as plt


class Word2Vec(Model):
    def __init__(self,
                 vocab_size,
                 embedding_dim):
        """
        initialization
        :param vocab_size:
        :param max_len:
        """
        super(Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.hidden_layer = keras.layers.Dense(self.embedding_dim,
                                               use_bias=False, name='dense')
        self.output_layer = keras.layers.Dense(self.vocab_size,
                                               use_bias=False,
                                               activation='softmax',
                                               name='output')

    def call(self, inputs):
        hidden = self.hidden_layer(inputs)
        output = self.output_layer(hidden)

        return output


def data_generator(skip_gram, vocab_size):
    """
    generate train data
    :param skip_gram: corpus
    :param vocab_size: length of word dict
    :return:
    """
    train_x =[]
    train_y = []
    for word in skip_gram:
        xi = np.zeros(vocab_size).tolist()
        xi[word[0]] = 1
        train_x.append(xi)
        yi = np.zeros(vocab_size).tolist()
        yi[word[1]] = 1
        train_y.append(yi)

    return train_x, train_y


if __name__ == '__main__':

    sentences = ['apple banana fruit banana orange fruit orange banana fruit',
                 'dog cat animal cat monkey animal monkey dog animal']

    word_list = ' '.join(sentences).split()
    vocab = list(set(word_list))
    word_dic = {w: i for i, w in enumerate(vocab)}
    vocab_size = len(vocab)

    skip_grams = []
    for sentence in sentences:
        words = sentence.split()
        for i in range(1, len(words)-1):
            head_word = word_dic[words[i]]
            context_word = [word_dic[words[i-1]], word_dic[words[i+1]]]
            for context in context_word:
                skip_grams.append([head_word, context])

    x, y = data_generator(skip_grams, vocab_size)
    model = Word2Vec(vocab_size, 10)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.CategoricalCrossentropy(), metrics=[
            'accuracy'])
    model.fit(x, y, epochs=1000, batch_size=5)

    for i, label in enumerate(vocab):
        x = model.get_layer('dense').get_weights()[0][i]

        print(label, ": ", x)
