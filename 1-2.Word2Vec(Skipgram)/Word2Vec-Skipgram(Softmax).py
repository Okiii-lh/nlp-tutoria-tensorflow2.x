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


class Word2Vec(Model):
    def __init__(self,
                 vocab_size,
                 embedding_dim):
        """
        initialization
        :param vocab_size:
        :param max_len:
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.hidden_layer = keras.layers.Dense(embedding_dim, use_bias=False)
        self.output_layer = keras.layers.Dense(vocab_size, use_bias=False)

    def call(self, inputs):
        hidden = self.hidden_layer(inputs)
        output = self.output_layer(hidden)

        return output


def data_generator(batch_size, skip_gram):
    """
    generate data
    :param batch_size: batch size
    :param skip_gram: corpus
    :return:
    """


if __name__ == '__main__':

    sentences = ['apple banana fruit banana orange fruit orange banana fruit',
                 'dog cat animal cat monkey animal monkey dog animal']

    word_list = ' '.join(sentences).split()
    vocab = list(set(word_list))
    word_dic = {w: i for i, w in enumerate(vocab)}
    vocab_size = len(vocab)

    print(word_dic)
    skip_grams = []
    for sentence in sentences:
        words = sentence.split()
        for i in range(1, len(words)-1):
            head_word = word_dic[words[i]]
            context_word = [word_dic[words[i-1]], word_dic[words[i+1]]]
            for context in context_word:
                skip_grams.append([head_word, context])

    print(skip_grams)