import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, TimeDistributed, Activation

num_students = 1000
num_trials = 16
code_length = 8
num_blocks = 4
embeddings_dim = 8

def add_post_embeddings_layers(model):
    hidden_dim = 32
    model.add(LSTM(hidden_dim, return_sequences=True, 
                   input_shape=(num_trials, embeddings_dim)))
    # TODO: Is there a way to train the dense networks with separate weights?
    model.add(TimeDistributed(Dense(1)))
    model.add(Activation("sigmoid"))

def fit_model(X, Y):
    model = Sequential()
    add_post_embeddings_layers(model)
    model.compile(loss="binary_crossentropy", 
                  optimizer="Adam",
                  metrics=["accuracy"])
    model.summary()
    model.fit(X, Y, batch_size=64, epochs=5, validation_split=0.1)


if __name__=="__main__":
    X = np.random.random((num_students, num_trials, embeddings_dim))
    Y = np.random.randint(2, size=(num_students, num_trials, 1))
    fit_model(X, Y)
