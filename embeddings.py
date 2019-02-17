#!/usr/bin/python3
import numpy as np
import random

random.seed(0)

def load_data(npy_file):
    """ Load data from file.
    
    Returns:
    all_matrix (np.array): (num_asts, num_timestamps, num_blocks) data set
    """
    return np.load(npy_file)


def split_data(all_matrix):
    """ Split data set into 3 groups, training/dev/test set.
    
    Split them in axis=0.
    Randomly pick 1% of ASTs and assign them to dev set.
    Randomly pick another 1% of ASTS and assign them to test set.
    
    Parameters:
    all_matrix (np.array): (num_asts, num_timestamps, num_blocks) data set

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
    timestep.
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


def train_model(model, train_matrix):
    """ Train the model with the input matrix. """
    #TODO: this is an example code in Keras. We need to replace this with the
    # code we want to reproduce.
    #model.fit(data, one_hot_labels, epochs=10, batch_size=32)
    pass


if __name__=="__main__":
    # Input: (M x T x B) matrix representing blocks in all ASTs
    all_matrix = load_data("./data-created/q4_array_of_ast_matrices.npy")
    print(all_matrix.shape)

    # Split into training/dev/test set
    train_matrix, dev_matrix, test_matrix = split_data(all_matrix)
    print(train_matrix.shape)
    print(dev_matrix.shape)
    print(test_matrix.shape)

    # Create model
    model = create_model()

    # Train model
    train_model(model, train_matrix)
