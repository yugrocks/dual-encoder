from get_test_models import get_response_model,get_context_encoding,dot_similarity
import numpy as np
import pickle
import pandas as pd
import nltk
from keras.preprocessing.sequence import pad_sequences
from clean_string import clean_string
from dual_encoder import *


def load_responses():
    with open("data/zini_response_encodings.pkl" ,'rb') as file:
        encodings = pickle.load(file)
    df = pd.read_csv("data/chat.csv",encoding="latin1")
    responses = df.iloc[:,1].values
    context = df.iloc[:,0].values
    return context, responses, encodings

def load_embeddings():
    vocab_size = 160740
    embeddings = np.zeros((vocab_size+1, 50)) # initializing as all zeros
    vocab = dict() # a dict of words such that their values contain their indexes
    vocab["__pad__"] = 0 # word at index 0 is the padding token
    # the index 0 of embedding matrix will contain all zeros for padding
    with open("data/glove.6B.50d.w2vformat_2.txt",encoding='utf8') as file:
        idx = 1
        for line in file:
            values = line.split()
            vocab[values[0]] = idx # add the word in vocab
            embeddings[idx] = np.asarray(values[1:], dtype='float32')
            idx += 1
    return vocab, embeddings
    


context, responses, encodings = load_responses()
vocab, embeddings = load_embeddings()
response_model = get_response_model(embeddings)

def predict_response(sentence, encodings,topn=5):
    sentence = nltk.tokenize.word_tokenize(sentence.lower())
    for j in range(len(sentence)):
        if str(sentence[j]) in vocab:
            sentence[j] = vocab[sentence[j]]  # replace the word in context by its index
        else:
            sentence[j] = vocab["__unk__"]  # unknown word
    sentence = pad_sequences([sentence,], maxlen=28, padding='pre')
    encoding = get_context_encoding(response_model, sentence)
    similarities = dot_similarity(encoding, encodings) # array of similarities
    sortd = sorted(zip(similarities[0], responses), reverse=True)
    return sortd[0:topn]

"""now predicting each and every sentence in responses and getting to know whether each response is the
correct one for given context. If it's not, then put this and others down, and if it is, then  carry on"""

X_context = []
X_response = []
y_similarities = []
max_candidates = 5

# iterate through each sentence in content
for i in range(len(context)):
    candidate_responses = predict_response(context[i], encodings,topn=max_candidates)
    # now check if the first response is the desired response
    if candidate_responses[0][1] == responses[i]:
        continue # then do nothing
    # else record the instances
    # first append the main context and response
    X_context.append(nltk.tokenize.word_tokenize(context[i]))
    X_response.append(nltk.tokenize.word_tokenize(responses[i]))
    y_similarities.append(1) # mark them similar
    # now iterate over the candidate_responses
    for res in candidate_responses:
        if clean_string(res[1].lower()).strip() == clean_string(responses[i].lower()).strip():
            # then do nothing
            continue
        else: # mark them as dissimilar
            X_context.append(nltk.tokenize.word_tokenize(context[i]))
            X_response.append(nltk.tokenize.word_tokenize(res[1]))
            y_similarities.append(0)


# now converting context and response vectors to integer sequences
for i in range(len(X_context)):
    for j in range(len(X_context[i])):
        if str(X_context[i][j]) in vocab:
            X_context[i][j] = vocab[X_context[i][j]]  # replace the word in context by its index
        else:
            X_context[i][j] = vocab["__unk__"]  # unknown word

for i in range(len(X_response)):
    for j in range(len(X_response[i])):
        if str(X_response[i][j]) in vocab:
            X_response[i][j] = vocab[X_response[i][j]]  # replace the word in context by its index
        else:
            X_response[i][j] = vocab["__unk__"]  # unknown word
print("DATASET AND EMBEDDINGS PREPARED")


# now pad them
X_context = pad_sequences(X_context, maxlen=28, padding='pre')
X_response = pad_sequences(X_response, maxlen=28, padding='pre')





#dual_encoder_model = get_dual_encoder(embeddings)
#dual_encoder_model.load_weights("weights.h5")
#
#
#dual_encoder_model.fit([X_context,X_response],y_similarities, batch_size=64,epochs=12,initial_epoch=11)
#
## save the weights again
#dual_encoder_model.save_weights("weights2.h5",overwrite=True)
## now save layer by layer
#for layer in dual_encoder_model.layers:
#    name = layer.name + "2"
#    with open("output/"+str(name), "wb") as file:
#        pickle.dump(layer.get_weights(), file)


