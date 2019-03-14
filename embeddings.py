#!/usr/bin/python3
import numpy as np
import random
import tensorflow as tf
import keras.models as models
import keras.layers as layers
import matplotlib.pyplot as plt

import utils

random.seed(0)

def create_model(num_timestep, num_blocks):
    """ Returns LSTM model for predicting next block in a given AST and a
    timestep
    """
    hidden_size = 128
    dropout_p = 0.5
    
    model = models.Sequential()
    
    #Add LSTM layer with 128 hidden units, tanh nonlinearity
    model.add(layers.LSTM(hidden_size, 
                          activation='tanh',
                          return_sequences=True,
                          input_shape=(num_timestep, num_blocks)))
    
    #Add Dropout
    #What about rescalling?, we shouuld add scale up to avoid modifying it during test time
    model.add(layers.Dropout(dropout_p))
    
    #Add Dense layer
    model.add(layers.Dense(num_blocks,
                           activation='softmax'))
    
    #Configure the learning process
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    
    model.summary()
    return model


def train_model(model, train_matrix, train_labels):
    """ Train the model with the input matrix. """
    
    #model.fit(train_matrix, train_labels, epochs=5)
    
    history = model.fit(train_matrix, train_labels, 
                        validation_split=0.1, 
                        epochs=50, 
                        batch_size=16, 
                        verbose=1)

    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def test_nn_framework():
    # Input: (M x T x B) matrix representing blocks in all ASTs
    all_matrix = utils.load_data("./data-created/q4_array_of_ast_matrices.npy")
    validation_matrix = utils.create_validation_matrix(all_matrix)
    print(all_matrix.shape)
    print(validation_matrix.shape)

    # Split into training/dev/test set
    train_matrix, dev_matrix, test_matrix = utils.split_data(all_matrix)
    print(train_matrix.shape)
    print(dev_matrix.shape)
    print(test_matrix.shape)

    # Create model
    num_timesteps = all_matrix.shape[1]
    num_blocks = all_matrix.shape[2]
    model = create_model(num_timesteps, num_blocks)

    # train_matrix model
    #train_matrix = all_matrix[0, :, :]
    #train_labels = validation_matrix[0, :, :]    
    #train_model(train_matrix, train_labels)
    train_model(model, all_matrix, validation_matrix)
    
    print(utils.accuracy_from_onehot_matrix(
              model.predict(test_matrix),
              utils.create_validation_matrix(test_matrix)))

if __name__=="__main__":
    test_nn_framework()
