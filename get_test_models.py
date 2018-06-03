from keras.layers.recurrent import LSTM
from keras.layers import merge, Dense, Activation, Embedding, Input
from keras.models import Sequential, Model
import pickle
import numpy as np


def get_response_model(embeddings):
    # returns a vector encoding of the response sentence
    encoder = Sequential(name="encoder")
    encoder.add(Embedding(embeddings.shape[0], 50,input_length=28,weights=[embeddings],trainable=False,mask_zero=True))
    encoder.add(LSTM(256, name="lstm1_encoder"))
    encoder.compile(loss='binary_crossentropy',optimizer='adam')
    # load weights
    with open("output/encoder5",'rb') as file:
        encoder.set_weights(pickle.load(file))
    return encoder


def get_context_encoding(encoder_model, context):
    vec1 = encoder_model.predict(context) # this is (1,256)
    with open("output/my_matrix_layer5",'rb') as file:
        my_matrix = pickle.load(file)  # this is (256, 256)
    # matrix multiplication
    vec2 = np.matmul(vec1,my_matrix[0])
    return vec2


def dot_similarity(vec1, vec2):
    dot = np.matmul(vec1, vec2.T)
    return (1 / (1 + np.exp(-dot)))
