import tensorflow as tf
import keras
import numpy as np
from keras import imdb                      #????????

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

print(train_data[0])

word_index =  imdb.get_word_index()         #????????

word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reversed_word_index = dict([(value, key)for (key, value) in word_index.item()])


def decode_review(text):                                                                    #returns readable text from train_data
    return " ".join([reversed_word_index.get(i, "?") for i in text])

print(decode_review(test_data[0]))


#unfinished