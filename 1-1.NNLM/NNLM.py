# -*- coding:utf-8 -*-
"""
 @Time: 2021/7/5 下午3:44
 @Author: LiuHe
 @File: NNLM.py
 @Describe: Nerual Network Language Model
"""
from tensorflow.keras import Model
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer


class NNLM(Model):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 max_len,
                 class_num):
        """
        initialization
        :param vocab_size:
        :param embedding_dim:
        :param max_len:
        :param class_num:
        """
        super(NNLM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.class_num = class_num

        self.embedding = keras.layers.Embedding(self.vocab_size,
                                                self.embedding_dim,
                                                input_length=self.max_len)
        self.LSTM = keras.layers.LSTM(128, return_sequences=False)
        self.out = keras.layers.Dense(self.vocab_size, activation='softmax')

    def call(self, inputs):
        embedding = self.embedding(inputs)
        lstm = self.LSTM(embedding)
        output = self.out(lstm)

        return output


def generate_data(sentences, max_len, vocab_size, tokenizer):
    """
    data generator
    :param sentences: corpus
    :param max_len:
    :param vocab_size:
    :param tokenizer:
    :return:
    """
    for sentence in sentences:
        words = sentence.split()
        inputs = [[tokenizer.word_index[n] for n in words[:-1]]]
        target = tokenizer.word_index[words[-1]]

        y = keras.utils.to_categorical([target], vocab_size)
        inputs_sequence = sequence.pad_sequences(inputs, maxlen=max_len)
        yield (inputs_sequence, y)


if __name__ == '__main__':
    sentences = ['i like dogs', 'i love coffee', 'i hate milk']

    tokenizer = Tokenizer()
    words = [word for words in sentences for word in words.split()]
    tokenizer.fit_on_texts(words)
    sequences = tokenizer.texts_to_sequences(words)

    vocab_size = len(tokenizer.word_index)+1
    embedding_dim = 128
    max_len = 4

    model = NNLM(vocab_size, embedding_dim, max_len, vocab_size)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')

    for x, y in generate_data(sentences, max_len, vocab_size, tokenizer):
        model.fit(x, y, epochs=1000)
