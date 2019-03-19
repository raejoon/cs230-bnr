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

tf.reset_default_graph()
keras.backend.clear_session()

class History(keras.callbacks.Callback):
    def __init__(self, X_train, Y_train):
        super().__init__()
        self.X_train = X_train
        self.Y_train = Y_train
        self.recall_train = {}
        self.recall_validate = {}
        self.epoch = 1

    def _get_recall(self, Y_true, Y_pred, p):
        P = 0.5 ** p
        num_students, num_asts = np.shape(Y_true)

        Y_true_flat = Y_true.flatten()
        Y_pred_flat = 1 * (Y_pred.flatten() > P)
        
        positive_mask = (Y_true_flat == 1.)
        if not np.any(positive_mask):
            return 0

        return np.mean(Y_pred_flat[positive_mask])
    
    def on_epoch_end(self, epoch, logs={}):
        X = self.X_train
        Y_true = self.Y_train
        Y_pred = self.model.predict(X)

        for p in np.arange(1, 4):
            self.recall_train[(p, self.epoch)] = \
                    self._get_recall(Y_true, Y_pred, p)

        X = self.validation_data[0]
        Y_true = self.validation_data[1]
        Y_pred = self.model.predict(X)

        for p in np.arange(1, 4):
            self.recall_validate[(p, self.epoch)] = \
                    self._get_recall(Y_true, Y_pred, p)

        print(self.recall_train)
        print(self.recall_validate)
        self.epoch += 1


def plot_recall_curves(history):
    for p in np.arange(1, 4):
        P = 0.5 ** p
        y_train = [history.recall_train[(p, e)] for e in range(1, history.epoch)]
        print(y_train)
    
        plt.plot(y_train, label="p_threshold: " + str(P))
    
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Test set recall")


def _get_input_matrix(secret_to_traj_map, traj_to_ast_map, maxlen):
    # TODO: need to have a better fix than getting rid of 272 entries
    X = np.zeros((len(secret_to_traj_map), maxlen))
    row_ind = 0
    for sid in sorted(secret_to_traj_map.keys()):
        tid = secret_to_traj_map[sid]
        real_part = traj_to_ast_map[tid]
        if len(real_part) == 0:
            continue
        appended_part = [real_part[-1]] * (maxlen - len(real_part))
        X[row_ind,:] = np.array(real_part + appended_part)
        row_ind += 1;
    return X + 1    # + 1 for 1-based indexing


def load_trajectories_from_dataset(trajectory_dirpath):
    idmap_file = os.path.join(trajectory_dirpath, "idMap.txt")
    # Get trajectory of each student
    secret_to_traj_map = {}
    with open(idmap_file) as fo:
        fo.readline()
        for line in fo:
            row = line.rstrip().split(",") 
            secret_id = int(row[0])
            traj_id = int(row[1])
            secret_to_traj_map[secret_id] = traj_id
        traj_to_ast_map = {tid: [] for tid in
                                set(secret_to_traj_map.values())}

    # Get ast list of each trajectory
    maxlen = 0
    maxtid = None
    empty_traj = None
    for tid in traj_to_ast_map:
        trajectory_file = os.path.join(trajectory_dirpath, str(tid) + ".txt")
        with open(trajectory_file) as fo: 
            ast_list = [int(line.strip()) for line in fo]
            if len(ast_list) == 0:
                empty_traj = tid
            traj_to_ast_map[tid] = ast_list
            ast_len = len(traj_to_ast_map[tid])
            if ast_len > maxlen:
                maxlen = ast_len
                maxtid = tid

    # Get rid of secret ids which maps to empty trajectory
    empty_sids = \
        [sid for sid, tid in secret_to_traj_map.items() if tid == empty_traj]
    for sid in empty_sids:
        del secret_to_traj_map[sid]
    
    # Construct matrix
    return _get_input_matrix(secret_to_traj_map, traj_to_ast_map, maxlen)

def load_trajectories_from_file(trajectory_filepath):
    return np.load(trajectory_filepath)

def save_trajectories_to_file(X, trajectory_filepath):
    np.save(trajectory_filepath, X)

def get_output_labels(X):
    pass

def load_output_labels_npy(output_filepath):
    Y = np.load(output_filepath)
    return Y

### Do not use these !!!! ###
"""
def save_output_labels_npy(Y, output_filepath):
    np.save(output_filepath, 
            np.reshape(Y, (np.shape(Y)[0], np.shape(Y)[1], 1)))

def save_output_labels_csv(Y, output_csvpath):
    with open(output_csvpath, "w") as fo: 
        for r in range(np.shape(Y)[0]):
            fo.write(str(r) + "," + ",".join(str(y) for y in Y[r,:]) + "\n");

def load_output_labels_csv(output_csvpath):
    with open(output_csvpath) as fo:
        data = [[int(e) for e in line.strip().split(",")[1:]] for line in fo]

    Y = np.array(data)
    return np.reshape(Y, (np.shape(Y)[0], np.shape(Y)[1], 1))
"""

def create_baseline_model(X, embeddings_dim, embeddings_matrix=None):
    ast_dirpath = "anonymizeddata/data/hoc4/asts"
    #num_asts = max(utils.get_ast_ids(ast_dirpath)) + 1
    num_asts = np.amax(X)
    num_trials = np.shape(X)[1]
    model = Sequential() 
    if embeddings_matrix is None:
        embedding_layer = Embedding(num_asts + 1, embeddings_dim, 
                                    input_length=num_trials)
    else:
        assert(embeddings_dim == np.shape(embeddings_matrix)[1])
        embedding_layer = Embedding(num_asts + 1, embeddings_dim, 
                                    input_length=num_trials, trainable=False)
        embedding_layer.build((None,))
        embedding_layer.set_weights([embeddings_matrix])

    model.add(embedding_layer)
    model.add(TimeDistributed(Dense(1)))
    model.add(Activation("sigmoid"))
    model.add(Reshape((num_trials,)))
    model.compile(loss="binary_crossentropy", optimizer="adam",
                  metrics=["accuracy", km.binary_recall()])
    model.summary()
    return model


def create_nn_model(X, embeddings_dim, embeddings_matrix=None):
    ast_dirpath = "anonymizeddata/data/hoc4/asts"
    num_asts = max(utils.get_ast_ids(ast_dirpath)) + 1
    num_trials = np.shape(X)[1]

    model = Sequential()

    if embeddings_matrix is None:
        embedding_layer = Embedding(num_asts + 1, embeddings_dim, 
                                    input_length=num_trials)
    else:
        assert(embeddings_dim == np.shape(embeddings_matrix)[1])
        embedding_layer = Embedding(num_asts + 1, embeddings_dim, 
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


def fit_model(model, X, Y, epochs=5):
    split_ind = int((1 - 0.1) * np.shape(X)[0])
    X_train = X[np.arange(split_ind), :]
    Y_train = Y[np.arange(split_ind), :]
    X_validate = X[np.arange(split_ind, np.shape(X)[0]), :]
    Y_validate = Y[np.arange(split_ind, np.shape(Y)[0]), :]

    history = model.fit(X_train, Y_train, 
                        validation_data=(X_validate, Y_validate), 
                        epochs=epochs, 
                        batch_size=16, 
                        verbose=1)

    return history, X_train, Y_train, X_validate, Y_validate


def _get_recall_from_vectors(y_true, y_pred):
    positive_mask = (y_true == 1.)
    if not np.any(positive_mask):
        return 0
    return np.mean(y_pred[positive_mask])


def plot_timestep_recall(model, X_train, Y_train, X_validate, Y_validate):
    Y_true = Y_train
    Y_pred = model.predict(X_train)
    Y_pred = 1 * (Y_pred > 0.5)

    recalls = [_get_recall_from_vectors(Y_true[:, ind], Y_pred[:, ind]) \
                    for ind in range(np.shape(X_train)[1])]
    print(recalls)
    plt.plot(recalls, label="Training set")

    Y_true = Y_validate
    Y_pred = model.predict(X_validate)
    Y_pred = 1 * (Y_pred > 0.5)

    recalls = [_get_recall_from_vectors(Y_true[:, ind], Y_pred[:, ind]) \
                    for ind in range(np.shape(X_validate)[1])]
    print(recalls)
    plt.plot(recalls, label="Validation set")
    plt.legend()
    plt.xlabel("Timesteps")
    plt.ylabel("Recall")
    plt.ylim((0, 1.02))

    plt.savefig("recall_over_time.png") 

    

