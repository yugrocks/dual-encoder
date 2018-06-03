import numpy as np
from sklearn.utils import shuffle
import pickle


# encoder input example :- pad pad...pad hello how are you <eos> 
# decoder input example :-  <SOS> I am fine , thankyou <eos> ...pad
path_input = r"data/"
class MasterGenerator:
    
    def __init__(self, X,y,w2v, batch_size=64, shuffle_dataset=True,seq_length=20): # takes in a gensim w2v instance
        if shuffle_dataset:
            self.X, self.y = shuffle(X,y,random_state=2)
        else:
            self.X = X; self.y = y
        self.batch_size = batch_size
        self.w2v = w2v
        self.max_length = seq_length
        self.m = len(X)
        with open(path_input+'unk','rb') as file:
            self.unk = pickle.load(file)['unk']
        self.unk = np.reshape(self.unk,(50,))
        
    def vectorize(self, word):
        flag = word in self.w2v
        if flag:
            return self.w2v[word]
        else:
            return self.unk
    
    def process(self, X,y):
        # to take each sentence in X and y and convert it to a series of vectors
        encoder_input = []
        decoder_output = []
        for i in range(len(X)):
            X_i = []
            y_i = []
            ln = min(self.max_length, len(X[i]))
            for j in range(self.max_length - ln-1): # pad sequences upto the length
                X_i.append(np.zeros((50,)))
            for j in range(ln):
                word = X[i][j]
                X_i.append(self.vectorize(word))
            if len(X_i) == 20:
                X_i[-1] = self.vectorize("EOS")
            else:
                X_i.append(self.vectorize("EOS"))
            encoder_input.append(X_i)
            # now decoder output
            for j in range(len(y[i])):
                word = y[i][j]
                y_i.append(self.vectorize(word))
            for j in range(self.max_length - len(y[i])):
                y_i.append(self.vectorize("EOS"))
            decoder_output.append(y_i)
        return encoder_input, decoder_output
    
    def get_generator(self):
        index = 0
        while True:
            X = []; y = []
            if index+self.batch_size > self.m:
                X.extend(self.X[index:self.m]); X.extend(self.X[0:index+self.batch_size-self.m])
                y.extend(self.y[index:self.m]); y.extend(self.y[0:index+self.batch_size-self.m])
                index = index+self.batch_size-self.m
            else:
                X = self.X[index : index+self.batch_size]
                y = self.y[index : index+self.batch_size]
                index += self.batch_size
            encoder_input,decoder_output = self.process(X, y)
            encoder_input = np.array(encoder_input)
            decoder_output = np.array(decoder_output)
            encoder_input = np.reshape(encoder_input, (self.batch_size, self.max_length, 50))            
            decoder_output = np.reshape(decoder_output, (self.batch_size, self.max_length, 50))
            yield encoder_input, decoder_output
