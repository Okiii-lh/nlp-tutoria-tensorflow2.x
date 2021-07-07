# -*- coding:utf-8 -*-
"""
 @Time: 2021/7/7 下午1:59
 @Author: LiuHe
 @File: TextCNN.py
 @Describe:
"""
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt


class TextCNN(Model):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 max_len,
                 kernel_sizes=[3, 4, 5],
                 class_num=2):
        """
        initialization
        :param vocab_size:
        :param embedding_dim:
        :param max_len:
        :param kernel_sizes:
        :param class_num:
        """
        super(TextCNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.kernel_sizes = kernel_sizes
        self.class_num = class_num

        self.embedding = keras.layers.Embedding(self.vocab_size,
                                                self.embedding_dim,
                                                input_length=self.max_len)
        self.convs = []
        self.max_poolings = []
        for kernel_size in self.kernel_sizes:
            self.convs.append(
                keras.layers.Conv1D(128, kernel_size, activation='relu')
            )
            self.max_poolings.append(
                keras.layers.GlobalAveragePooling1D()
            )

        self.classifier = keras.layers.Dense(self.class_num,
                                             activation='sigmoid')

    def call(self, inputs):
        embedding = self.embedding(inputs)
        convs = []
        for i in range(len(self.convs)):
            x = self.convs[i](embedding)
            x = self.max_poolings[i](x)
            convs.append(x)
        x = keras.layers.Concatenate()(convs)
        output = self.classifier(x)

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

    model = TextCNN(vocab_size, embedding_dim, max_length)
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