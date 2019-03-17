import os
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, TimeDistributed, Activation


def _get_input_matrix(trajectory_list, traj_to_ast_map, maxlen):
    X = np.zeros((len(trajectory_list), maxlen))
    for row_ind, tid in enumerate(trajectory_list):
        real_part = traj_to_ast_map[tid]
        appended_part = [real_part[-1]] * (maxlen - len(real_part))
        X[row_ind,:] = np.array(real_part + appended_part)
    return X


def load_trajectories_from_dataset(trajectory_dirpath):
    idmap_file = os.path.join(trajectory_dirpath, "idMap.txt")
    # Get trajectory of each student
    with open(idmap_file) as fo:
        fo.readline()
        trajectory_list = [int(line.strip().split(",")[1]) for line in fo]
        traj_to_ast_map = {tid: [] for tid in set(trajectory_list)}
    
    # Get ast list of each trajectory
    maxlen = 0
    for tid in traj_to_ast_map:
        trajectory_file = os.path.join(trajectory_dirpath, str(tid) + ".txt")
        with open(trajectory_file) as fo: 
            traj_to_ast_map[tid] = [int(line.strip()) for line in fo]
            maxlen = max(maxlen, len(traj_to_ast_map[tid]))
    
    # Construct matrix
    return _get_input_matrix(trajectory_list, traj_to_ast_map, maxlen)


def load_trajectories_from_file(trajectory_filepath):
    return np.load(trajectory_filepath)

def save_trajectories_to_file(X, trajectory_filepath):
    np.save(trajectory_filepath, X)

def get_output_labels(X):
    pass

def load_output_labels_npy(output_filepath):
    return np.load(output_filepath)

def save_output_labels_npy(Y, output_filepath):
    np.save(output_filepath, Y)

def save_output_labels_csv(Y, output_csvpath):
    with open(output_csvpath, "w") as fo: 
        for r in range(np.shape(Y)[0]):
            fo.write(str(r) + "," + ",".join(str(y) for y in Y[r,:]) + "\n");

def load_output_labels_csv(output_csvpath):
    with open(output_csvpath) as fo:
        data = [[int(e) for e in line.strip().split(",")[1:]] for line in fo]
    
    return np.array(data)


def create_baseline_model(X, num_asts, embeddings_dim, embeddings_matrix=None):
    num_trials = np.shape(X)[1]
    model = Sequential() 
    if embeddings_matrix is None:
        embedding_layer = Embedding(num_asts+1, embeddings_dim, 
                                    input_length=num_trials)
    else:
        assert(embeddings_dim == np.shape(embeddings_matrix)[1])
        embedding_layer = Embedding(num_asts+1, embeddings_dim, 
                                    input_length=num_trials, trainable=False)
        embedding_layer.build((None,))
        embedding_layer.set_weights([embeddings_matrix])

    model.add(embedding_layer)
    model.add(TimeDistributed(Dense(1)))
    model.add(Activation("sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam",
                  metrics=["accuracy"])
    model.summary()
    return model


def create_nn_model(X, num_asts, embeddings_dim, embeddings_matrix=None):
    num_trials = np.shape(X)[1]

    model = Sequential()

    if embeddings_matrix is None:
        embedding_layer = Embedding(num_asts+1, embeddings_dim, 
                                    input_length=num_trials)
    else:
        assert(embeddings_dim == np.shape(embeddings_matrix)[1])
        embedding_layer = Embedding(num_asts+1, embeddings_dim, 
                                    input_length=num_trials, trainable=False)
        embedding_layer.build((None,))
        embedding_layer.set_weights([embeddings_matrix])

    model.add(embedding_layer)
    
    hidden_dim = 32
    model.add(LSTM(hidden_dim, return_sequences=True, 
                   input_shape=(num_trials, embeddings_dim)))
    # TODO: Is there a way to train the dense networks with separate weights?
    model.add(TimeDistributed(Dense(1)))
    model.add(Activation("sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam",
                  metrics=["accuracy"])
    model.summary()
    return model


def fit_model(model, X, Y):
    model.fit(X, Y, batch_size=64, epochs=5, validation_split=0.1)
