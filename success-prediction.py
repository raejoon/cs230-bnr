import itertools
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, TimeDistributed, Activation

num_students = 1000
num_trials = 16
code_length = 8
num_blocks = 4
embeddings_dim = 8

code_vocab_size = num_blocks ** code_length


def code_hash(block_array):
    return np.dot(block_array, num_blocks ** np.arange(code_length))


def add_post_embedding_layers(model):
    hidden_dim = 32
    model.add(LSTM(hidden_dim, return_sequences=True, 
                   input_shape=(num_trials, embeddings_dim)))
    # TODO: Is there a way to train the dense networks with separate weights?
    model.add(TimeDistributed(Dense(1)))
    model.add(Activation("sigmoid"))


def create_model(embedding_matrix=None):
    model = Sequential()
    
    if embedding_matrix is None:
        embedding_layer = Embedding(code_vocab_size+1, embeddings_dim, 
                                    input_length=num_trials)
    else:
        embedding_layer = Embedding(code_vocab_size+1, embeddings_dim, 
                                    input_length=num_trials, trainable=False)
        embedding_layer.build((None,))
        embedding_layer.set_weights([embedding_matrix])

    model.add(embedding_layer)
    add_post_embedding_layers(model)
    model.compile(loss="binary_crossentropy", optimizer="adam",
                  metrics=["accuracy"])
    model.summary()
    return model

def visualize(history):
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


if __name__=="__main__":
    X = np.random.random((num_students, num_trials, code_length))
    X_hashed = np.zeros((num_students, num_trials))
    for i, j in itertools.product(range(num_students), range(num_trials)):
        X_hashed[i, j] = code_hash(X[i, j, :])

    Y = np.random.randint(2, size=(num_students, num_trials, 1))
    model = create_model()
    history = \
        model.fit(X_hashed, Y, batch_size=64, epochs=5, validation_split=0.1)

    #visualize(history)
