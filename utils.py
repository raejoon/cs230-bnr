#!/usr/bin/python3

import numpy as np
import random

def load_data(npy_file):
    """ Load data from file.
    Note that npy file generated by Ben's code has the following shape.
    (num_timsteps, num_blocks, num_asts)
    
    Returns:
    all_matrix (np.array): (num_asts, num_timesteps, num_blocks) data set
    """
    orig_mat = np.load(npy_file)
    orig_mat = np.swapaxes(orig_mat, 0, 1)
    orig_mat = np.swapaxes(orig_mat, 0, 2)
    return orig_mat


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


def onehot_encode(array, num_classes):
    """ Encode one hot vectors from a 1D array. """
    n_entries = len(array)
    array = array.astype(int)
    onehot_matrix = np.zeros((n_entries, num_classes))
    onehot_matrix[np.arange(n_entries), array] = 1
    return onehot_matrix


def onehot_decode(onehot_matrix):
    """ Decode one hot row vectors to a column vector of class ids. """
    nrow, ncol = np.shape(onehot_matrix)
    range_vector = np.reshape(np.arange(ncol), (ncol, 1))
    return np.matmul(onehot_matrix, range_vector)


def accuracy_from_class_matrix(prediction_matrix, validation_matrix, eoa_ind):
    """ Calculate accuracy from two class matrices.
        TODO: still inaccurate since figuring out the first occurence of eoa
        block is not trivial but the current code discards the outcome.
    """
    num_asts, num_timesteps = np.shape(prediction_matrix)
    pred_vec = prediction_matrix.flatten()
    vali_vec = validation_matrix.flatten()
    pred_vec_filtered = pred_vec[vali_vec != eoa_ind]
    vali_vec_filtered = vali_vec[vali_vec != eoa_ind]
    correct = np.sum(1 * (pred_vec_filtered != vali_vec_filtered))
    return correct / len(vali_vec_filtered)


def accuracy_from_onehot_matrix(prediction_matrix, validation_matrix):
    """ Calculate accuracy from two onehot encoded matrices. """
    _, _, num_blocks = np.shape(prediction_matrix)
    pred_class_matrix = np.argmax(prediction_matrix, axis=-1)
    vali_class_matrix = np.argmax(validation_matrix, axis=-1)
    return accuracy_from_class_matrix(pred_class_matrix, vali_class_matrix,
                                      eoa_ind=num_blocks-1)
