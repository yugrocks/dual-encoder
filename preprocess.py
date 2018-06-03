import numpy as np
import nltk
import pandas as pd
from clean_string import clean_string


path_input = r"data/"
path_output = r"output/"
nltk.download('punkt')
print("PREPARING DATA AND EMBEDDINGS")
sentence_length = 28


X = []
y = []

train = open(path_input+"chats.txt",encoding="utf8",errors='ignore')

write_responses = []
write_messages = []

for line in train:
    if line[0] == 'Q':
        response = line[3:]
        write_messages.append(response)
        X.append(clean_string(response.lower()).split())
    elif line[0] == 'A':
        response = line[3:]
        write_responses.append(response)
        y.append(clean_string(response.lower()).split())
    else:
        continue
train.close()

a = np.array([write_messages,write_responses]).T

df = pd.DataFrame(a, columns=["MESSAGE","RESPONSE"])
df.to_csv(path_output+"chat.csv")

for i in range(len(X)):
    if len(X[i]) > sentence_length:
        X[i] = X[i][0:sentence_length]
    if len(y[i]) > sentence_length:
        y[i] = y[i][0:sentence_length]
print("Data loaded.")


context = []  # the context message
response = [] # the response to be given
similarity = [] # tells if context and response are similar


for i in range(len(X)):
    # take the ith sentences and add them directly
    for d in range(5):  # duplicate the matching pair 5 times
        context.append(X[i].copy()) ; response.append(y[i].copy()) ; similarity.append(1)
    # now take 100 more random responses which do not fit with context
    for j in range(100):
        random_idx = np.random.randint(0,len(X)-1)
        if random_idx == i: # if random number comes equal to i
            random_idx += 1 # then add some number to it
        context.append(X[i].copy()) ; response.append(y[random_idx].copy()) ; similarity.append(0)


del X
del y
# Now we convert them to integer values
# First we load the Glove vectors file
vocab_size = 160740

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

# now converting context and response vectors to integer sequences
for i in range(len(context)):
    for j in range(len(context[i])):
        if str(context[i][j]) in vocab:
            context[i][j] = vocab[context[i][j]]  # replace the word in context by its index
        else:
            context[i][j] = vocab["__unk__"]  # unknown word

for i in range(len(response)):
    for j in range(len(response[i])):
        if str(response[i][j]) in vocab:
            response[i][j] = vocab[response[i][j]]  # replace the word in context by its index
        else:
            response[i][j] = vocab["__unk__"]  # unknown word
print("DATASET AND EMBEDDINGS PREPARED")
