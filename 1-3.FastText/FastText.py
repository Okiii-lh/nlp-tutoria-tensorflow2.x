# -*- coding:utf-8 -*-
"""
 @Time: 2021/7/7 下午1:29
 @Author: LiuHe
 @File: FastText.py
 @Describe:
"""
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
import matplotlib.pyplot as plt


class FastText(Model):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 max_len,
                 class_num):
        """
        initialization
        :param vocab_size: vocabulary
        :param embedding_dim: embedding dim
        :param max_len: the vector's length
        :param class_num: count of labels
        """
        super(FastText, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.class_num = class_num

        self.embedding = keras.layers.Embedding(self.vocab_size,
                                                self.embedding_dim,
                                                input_length=self.max_len)
        self.pooling = keras.layers.GlobalAveragePooling1D()
        self.classifier = keras.layers.Dense(self.class_num,
                                             activation='softmax')

    def call(self, inputs):
        embedding = self.embedding(inputs)
        pool = self.pooling(embedding)
        output = self.classifier(pool)

        return output


if __name__ == '__main__':
    train_x = ['i love you', 'he loves me', 'she likes baseball',
               'i hate you', 'sorry for that', 'this is awful']
    train_y = [1, 1, 1, 0, 0, 0]

    vocab_size = 10
    embedding_dim = 20
    max_len = 3
    pad_token = '<OOV>'
    class_num = 2
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=pad_token)
    tokenizer.fit_on_texts(train_x)
    train_x_sequences = tokenizer.texts_to_sequences(train_x)
    print(train_x_sequences)
    train_x_sequences = sequence.pad_sequences(train_x_sequences, max_len)
    print(train_x_sequences)

    train_y = keras.utils.to_categorical(train_y)
    print(train_y)

    model = FastText(vocab_size, embedding_dim, max_len, class_num)
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.CategoricalCrossentropy(), metrics=[
            'accuracy'])

    history = model.fit(train_x_sequences, train_y, epochs=30)


    def plot_graphs(history, string):
        plt.plot(history.history[string])
        plt.xlabel('epoches')
        plt.ylabel(string)
        plt.show()


    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')