### nlp-tutorial-tensorflow2.x

​	<center>![tf2](https://cdn.jsdelivr.net/gh/Okiii-lh/images/img/tf2-20210705145821081.png)</center>

nlp-tutorial-tensorflow2.x is a tutorial for who is studying NLP using TensorFlow2.x. Most of the model in NLP were implemented with less 100 lines of code(except comments or blank lines)

Reference: https://github.com/graykode/nlp-tutorial （implemented by Pytorch）

## Curriculum - (Example Purpose)

#### 1. Basic Embedding Model

- 1-1.NNLM(Neural Network Language Model) - **Predict Next Word**
  - Paper - [A Neural Probabilistic Language Model(2003)](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
  - Code - [NNLM.py](https://github.com/Okiii-lh/nlp-tutoria-tensorflow2.x/blob/master/1-1.NNLM/NNLM.py)
- 1-2.Word2Vec(Skip-gram) - **Embedding Words and Show Graph**
  - Paper - [Distributed Representations of Words and Phrases and their Compositionality(2013)](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
  - Code - [Word2Vec-Skipgram(Softmax).py](https://github.com/Okiii-lh/nlp-tutoria-tensorflow2.x/blob/master/1-2.Word2Vec(Skipgram)/Word2Vec-Skipgram(Softmax).py)
- 1-3.FastText(Application Level) - **Sentence Classification**
  - Paper - [Bag of Tricks for Efficient Text Classification(2016)](https://arxiv.org/pdf/1607.01759.pdf)
  - Code - [FastText.py](https://github.com/Okiii-lh/nlp-tutoria-tensorflow2.x/blob/master/1-3.FastText/FastText.py)

#### 2. CNN(Convolutional Neural Network)

- 2-1.TextCNN - **Binary Sentiment Classification**
  - Paper - [Convolutional Neural Networks for Sentence Classification(2014)](http://www.aclweb.org/anthology/D14-1181)
  - Code - [TextCNN.py](https://github.com/Okiii-lh/nlp-tutoria-tensorflow2.x/blob/master/2-1.TextCNN/TextCNN.py)

#### 3. RNN(Recurrent Neural Network)

- 3-1.TextRNN - **Sentiment Classification**
  - Paper - [Finding Structure in Time(1990)](http://psych.colorado.edu/~kimlab/Elman1990.pdf)
  - Code - [TextRNN.py](https://github.com/Okiii-lh/nlp-tutoria-tensorflow2.x/blob/master/3-1.TextRNN/TextRNN.py)
- 3-2.TextLSTM - **Autocomplete**
  - Paper - [LONG SHORT-TERM MEMORY(1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)
  - Code - TextLSTM.py
- 3-3.Bi-LSTM - **Predict Next Word in Long Sentence**
  - Code - waiting

#### 4. Attention Mechanism

- 4-1. Seq2Seq - **Change Word**
  - Paper - [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation(2014)](https://arxiv.org/pdf/1406.1078.pdf)
  - Code - waiting
- 4-2.Seq2Seq with Attention - **Translate**
  - Paper - [Neural Machine Translation by Jointly Learning to Align and Translate(2014)](https://arxiv.org/abs/1409.0473)
  - Code - waiting
- 4-3.Bi-LSTM with Attention - **Binary Sentiment Classification**
  - Code - waiting

#### 5. Model based on Transformer

- 5-1.The Transformer - **Translate**
  - Paper - [Attention Is All You Need(2017)](https://arxiv.org/abs/1706.03762)
  - Code - waiting
- 5-2.BERT - **Classification Next Sentence & Predict Masked Tokens**
  - Paper - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding(2018)](https://arxiv.org/abs/1810.04805)
  - Code - waiting

## Dependencies

- Python 3.7+
- TensorFlow 2.3.0

