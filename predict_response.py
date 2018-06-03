import pickle
import pandas as pd
import nltk
import numpy as np
from get_test_models import *
from keras.preprocessing.sequence import pad_sequences
from dual_encoder import get_dual_encoder
from clean_string import clean_string



class Yugalsbot:

    def __init__(self):
        print("LOADING RESPONSES")
        with open("data/VINI_response_encodings5.pkl" ,'rb') as file:
            self.encodings = pickle.load(file)
        df = pd.read_csv("data/chat.csv",encoding="latin1")
        self.responses = df.iloc[:,1].values
        print("RESPONSES LOADED")
        vocab_size = 160740
        self.embeddings = np.zeros((vocab_size+1, 50)) # initializing as all zeros
        self.vocab = dict() # a dict of words such that their values contain their indexes
        self.vocab["__pad__"] = 0 # word at index 0 is the padding token
        print("LOADING GLOVE EMBEDDINGS")
        # the index 0 of embedding matrix will contain all zeros for padding
        with open("data/glove.6B.50d.w2vformat_2.txt",encoding='utf8') as file:
            idx = 1
            for line in file:
                values = line.split()
                self.vocab[values[0]] = idx # add the word in vocab
                self.embeddings[idx] = np.asarray(values[1:], dtype='float32')
                idx += 1
        self.model = get_response_model(self.embeddings)
    
    def predict_response(self, sentence, topn=5):
        sentence = clean_string(sentence.lower()).split()
        for j in range(len(sentence)):
            if str(sentence[j]) in self.vocab:
                sentence[j] = self.vocab[sentence[j]]  # replace the word in context by its index
            else:
                sentence[j] = self.vocab["__unk__"]  # unknown word
        sentence = pad_sequences([sentence,], maxlen=28, padding='pre')
        encoding = get_context_encoding(self.model, sentence)
        print(encoding.shape)
        similarities = dot_similarity(encoding, self.encodings) # array of similarities
        sortd = sorted(zip(similarities[0], self.responses), reverse=True)
        return sortd[0:topn]


bot = Yugalsbot()

while True:
    message = input()
    if message == "q":
        break
    res = bot.predict_response(message)
    print("VINI:  ")
    for a,b in res:
        print(b," : ", a)

