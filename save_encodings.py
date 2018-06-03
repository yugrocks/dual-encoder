from get_test_models import *
from keras.preprocessing.sequence import pad_sequences
import nltk
import pandas as pd

path_input = "data/"

df = pd.read_csv(path_input+"chat.csv", encoding='latin1')

responses0 = df.iloc[:,1].values
responses = []

for i in range(len(responses0)):
    responses.append(nltk.tokenize.word_tokenize(responses0[i].lower()))



    
# Now we convert them to integer values
# First we load the Glove vectors file
vocab_size = 160740
embedding_length  = 50
encoding_length = 256

embeddings = np.zeros((vocab_size+1, 50)) # initializing as all zeros
vocab = dict() # a dict of words such that their values contain their indexes
vocab["__pad__"] = 0 # word at index 0 is the padding token


# the index 0 of embedding matrix will contain all zeros for padding
with open(path_input+"glove.6B.50d.w2vformat_2.txt",encoding='utf8') as file:
    idx = 1
    for line in file:
        values = line.split()
        vocab[values[0]] = idx # add the word in vocab
        embeddings[idx] = np.asarray(values[1:], dtype='float32')
        idx += 1

for i in range(len(responses)):
    for j in range(len(responses[i])):
        if str(responses[i][j]) in vocab:
            responses[i][j] = vocab[responses[i][j]]  # replace the word in context by its index
        else:
            responses[i][j] = vocab["__unk__"]  # unknown word
print("DATASET AND EMBEDDINGS PREPARED")


# functions get_response_model, get_context_encoding, dot_similarity are imported

encodings = np.zeros((len(responses), encoding_length))
model2 = get_response_model(embeddings)
responses = pad_sequences(responses, maxlen=28, padding='pre')


for i in range(responses.shape[0]):
    encoding = model2.predict(responses[i].reshape(1,28))
    encodings[i] = encoding


with open(path_input+"VINI_response_encodings5.pkl" ,'wb') as file:
    pickle.dump(encodings, file)
