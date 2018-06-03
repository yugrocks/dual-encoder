from keras.layers.recurrent import LSTM
from keras.layers import merge, Dense, Activation, Embedding, Input
from keras.models import Sequential, Model
from keras.optimizers import Adam


sentence_length = 28

def get_dual_encoder(embeddings):
    encoder = Sequential(name="encoder")
    encoder.add(Embedding(embeddings.shape[0], 50,input_length=sentence_length,weights=[embeddings],trainable=False,mask_zero=True))
    encoder.add(LSTM(256, name="lstm1_encoder"))
    context_input = Input(shape=(sentence_length,))
    response_input = Input(shape=(sentence_length,))
    context_branch = encoder(context_input)
    context_branch_end = Dense(256,use_bias=False, activation = None,name="my_matrix_layer") (context_branch)
    response_branch = encoder(response_input)
    concatenated = merge([context_branch_end, response_branch], mode='dot')
    out = Activation("sigmoid")(concatenated)
    dual_encoder = Model([context_input, response_input], out)
    adam = Adam(0.0005)
    dual_encoder.compile(loss='binary_crossentropy',
                    optimizer=adam,metrics=["accuracy"])
    dual_encoder.summary()
    return dual_encoder
