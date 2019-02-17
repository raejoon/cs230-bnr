#!/usr/bin/python3
import numpy as np
import random

random.seed(0)

def load_data(npy_file):
    """ Load data from file.
    
    Returns:
    all_matrix (np.array): (num_asts, num_timesteps, num_blocks) data set
    """
    return np.load(npy_file)


def create_validation_matrix(all_matrix):
    """ Create a matrix representing next block of each timestep of each AST.
    
    If data matrix is X and validation matrix is Y, Y[a,t,:] = X[a,t+1,:]
    The last timestep needs to predict the end of a AST, so will be
        [0, 0, ..., 0, 1]

    Returns:
    validation_matrix (np.array):
        (num_asts, num_timesteps, num_blocks) validation matrix
    """
    num_asts = all_matrix.shape[0]
    num_timesteps = all_matrix.shape[1]

    validation_matrix = np.zeros(all_matrix.shape)
    for t in range(num_timesteps - 1):
        validation_matrix[:,t,:] = all_matrix[:,t+1,:]

    validation_matrix[:,num_timesteps-1,-1] = np.ones(num_asts)
    return validation_matrix

def split_data(all_matrix):
    """ Split data set into 3 groups, training/dev/test set.
    
    Split them in axis=0.
    Randomly pick 1% of ASTs and assign them to dev set.
    Randomly pick another 1% of ASTS and assign them to test set.
    
    Parameters:
    all_matrix (np.array): (num_asts, num_timesteps, num_blocks) data set

    Returns:
    train_matrix (np.array)
    dev_matrix (np.array)
    test_matrix (np.array)
    """
    num_asts = np.shape(all_matrix)[0]  
    test_set_size = num_asts // 100
    
    data_indices = list(range(num_asts))
    random.shuffle(data_indices)
    
    train_matrix = all_matrix[2*test_set_size:]
    dev_matrix = all_matrix[:test_set_size]
    test_matrix = all_matrix[test_set_size:2*test_set_size]
    return train_matrix, dev_matrix, test_matrix


def create_model():
    """ Returns LSTM model for predicting next block in a given AST and a
    timestep
    """
    #TODO: this is an example code in Keras. We need to replace this with the
    # code we want to reproduce.
    #model = Sequential()
    #model.add(Dense(32, activation='relu', input_dim=100))
    #model.add(Dense(10, activation='softmax'))
    #model.compile(optimizer='rmsprop',
    #              loss='categorical_crossentropy',
    #              metrics=['accuracy'])
    return None


def create_dummy_model(num_timestep, num_blocks):
    """ Dummy model to work with. Remove this after merging code with
    Neeraj.
    """
    import keras.models as models
    model = models.Sequential()
    model.add(models.layers.LSTM(32, input_shape=(num_timestep, num_blocks)))
    model.add(Dense(num_blocks, activation="softmax"))
    
    model.compile(loss="categorical_crossentropy",
                  optimizer="rmsprop",
                  metrics=["accuracy"])
    return model


def train_model(model, train_matrix):
    """ Train the model with the input matrix. """
    #TODO: this is an example code in Keras. We need to replace this with the
    # code we want to reproduce.
    #model.fit(data, one_hot_labels, epochs=10, batch_size=32)
    pass


def train_dummy_model():
    pass


if __name__=="__main__":
    # Input: (M x T x B) matrix representing blocks in all ASTs
    all_matrix = load_data("./data-created/q4_array_of_ast_matrices.npy")
    validation_matrix = create_validation_matrix(all_matrix)
    print(all_matrix.shape)
    print(validation_matrix.shape)

    # Split into training/dev/test set
    train_matrix, dev_matrix, test_matrix = split_data(all_matrix)
    print(train_matrix.shape)
    print(dev_matrix.shape)
    print(test_matrix.shape)

    # Create model
    model = create_model()

    # Train model
    train_model(model, train_matrix)
