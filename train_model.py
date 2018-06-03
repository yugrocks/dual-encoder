from dual_encoder import get_dual_encoder
from preprocess import *
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle


path_output = "/output/"
path_input2 = "/data2/"


padded_context = pad_sequences(context, maxlen=28, padding='pre')
padded_responses = pad_sequences(response, maxlen=28, padding='pre')



model = get_dual_encoder(embeddings)
# optionally load weights. ONLY FOR RETRAINING
#model.load_weights("weights3.h5")
# loading data for encoder4 and my_matrix_layer4
encoder = model.get_layer("encoder")
with open("output/encoder",'rb') as wts:
    encoder.set_weights(pickle.load(wts))
my_matrix_layer = model.get_layer("my_matrix_layer")
with open("output/my_matrix_layer",'rb') as wts:
    my_matrix_layer.set_weights(pickle.load(wts))



# context, responses, similarity are already loaded
model.fit([padded_context,padded_responses],similarity, batch_size=1,epochs=20,initial_epoch=9)

model.save_weights("weights_dual_enc_model.h5",overwrite=True)
# now also save layer by layer
for layer in model.layers:
    name = layer.name+"dual_enc_model"
    with open(path_output+str(name), "wb") as file:
        pickle.dump(layer.get_weights(), file)





