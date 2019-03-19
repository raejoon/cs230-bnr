import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers \
    import LSTM, Dense, Embedding, TimeDistributed, Activation, Reshape
import utils
import keras_metrics as km
import matplotlib.pyplot as plt
import tensorflow as tf

def load_trajectories_from_file(trajectory_filepath):
    return np.load(trajectory_filepath)

def save_trajectories_to_file(X, trajectory_filepath):
    np.save(trajectory_filepath, X)

def load_output_labels_npy(output_filepath):
    Y = np.load(output_filepath)
    return Y

def create_baseline_model(X, embeddings_dim, embeddings_matrix=None):
    num_asts = np.amax(X)
    max_timestep = np.shape(X)[1]
    model = Sequential() 
    if embeddings_matrix is None:
        embedding_layer = Embedding(num_asts + 1, embeddings_dim, 
                                    input_length=max_timestep)
    else:
        embeddings_dim = np.shape(embeddings_matrix)[1]
        embedding_layer = Embedding(num_asts + 1, embeddings_dim, 
                                    input_length=max_timestep, trainable=False)
        embedding_layer.build((None,))
        embedding_layer.set_weights([embeddings_matrix])

    model.add(embedding_layer)
    model.add(TimeDistributed(Dense(1)))
    model.add(Activation("sigmoid"))
    model.add(Reshape((max_timestep,)))
    model.compile(loss="binary_crossentropy", optimizer="sgd",
                  metrics=["accuracy", km.binary_recall()])
    model.summary()
    return model

def create_nn_model(X, embeddings_dim, embeddings_matrix=None):
    num_asts = np.amax(X)
    max_timestep = np.shape(X)[1]

    model = Sequential()
    if embeddings_matrix is None:
        embedding_layer = Embedding(num_asts + 1, embeddings_dim, 
                                    input_length=max_timestep)
    else:
        embeddings_dim = np.shape(embeddings_matrix)[1]
        embedding_layer = Embedding(num_asts + 1, embeddings_dim, 
                                    input_length=max_timestep, trainable=False)
        embedding_layer.build((None,))
        embedding_layer.set_weights([embeddings_matrix])

    model.add(embedding_layer)
    
    hidden_dim = 32
    model.add(LSTM(hidden_dim, return_sequences=True, 
                   input_shape=(max_timestep, embeddings_dim)))
    model.add(TimeDistributed(Dense(1)))
    model.add(Activation("sigmoid"))
    model.add(Reshape((max_timestep,)))

    model.compile(loss="binary_crossentropy", optimizer="adam",
                  metrics=["accuracy", km.binary_recall()])
    model.summary()
    return model

def fit_model(model, data, epochs=5):
    X_train = data["X_train"]
    Y_train = data["Y_train"]
    X_validate = data["X_validate"]
    Y_validate = data["Y_validate"]
    model.fit(X_train, Y_train, validation_data=(X_validate, Y_validate), 
              epochs=epochs, 
              batch_size=16, 
              verbose=1)

def _get_recall_from_vectors(y_true, y_pred):
    positive_mask = (y_true == 1.)
    if not np.any(positive_mask):
        return 0
    return np.mean(y_pred[positive_mask])

def plot_timestep_recall(model, data):
    X_train = data["X_train"]
    Y_train = data["Y_train"]
    X_validate = data["X_validate"]
    Y_validate = data["Y_validate"]
    
    # Plot training set
    Y_true = Y_train
    Y_pred = model.predict(X_train)
    Y_pred = 1 * (Y_pred > 0.5)
    recalls = [_get_recall_from_vectors(Y_true[:, ind], Y_pred[:, ind]) \
                    for ind in range(np.shape(X_train)[1])]
    plt.plot(recalls, label="Training set")

    # Plot validation set
    Y_true = Y_validate
    Y_pred = model.predict(X_validate)
    Y_pred = 1 * (Y_pred > 0.5)
    recalls = [_get_recall_from_vectors(Y_true[:, ind], Y_pred[:, ind]) \
                    for ind in range(np.shape(X_validate)[1])]
    plt.plot(recalls, label="Validation set")
    
    plt.legend()
    plt.xlabel("Timesteps")
    plt.ylabel("Recall")
    plt.ylim((0, 1.02))
    plt.savefig("recall_over_time.png") 

    

