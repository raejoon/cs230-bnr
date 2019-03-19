import embeddings
import predictions
import utils
import numpy as np
import matplotlib.pyplot as plt

embed_dict_filename = "tmp/my_embeddings.npy"

def embeddings_run():
    ast_filepath = "processed/q4_array_of_ast_matrices.npy"
    embed_input = embeddings.load_asts_from_file(ast_filepath) 
    embed_output = embeddings.get_output_labels(embed_input)
    
    embed_model = embeddings.create_model(embed_input)
    embed_history = embeddings.fit_model(embed_model, 
                                         embed_input, embed_output,
                                         epochs=1)
    #print(embed_history.effective_accuracy["train"])
    #print(embed_history.effective_accuracy["validate"])

    embed_model_filename = "tmp/my_embeddings.h5"
    utils.save_model(embed_model, embed_model_filename)
    embed_model = utils.load_saved_model(embed_model_filename)
    
    ast_dirpath = "anonymizeddata/data/hoc4/asts/"
    embed_matrix = embeddings.get_embeddings(embed_model, embed_input,
                                           ast_dirpath)

    print("Embeddings matrix (including 1st row) size: ",
          np.shape(embed_matrix))
    embeddings.save_embeddings(embed_matrix, embed_dict_filename)    


def main():
    # Embeddings generation
    embeddings_run()
     
    # Failure prediction (Main task)
    input_filepath = "processed/traj_ast_matrix.npy"
    X = predictions.load_trajectories_from_file(input_filepath)

    output_filepath = "processed/traj_fail_window_2_matrix.npy"
    Y = predictions.load_output_labels_npy(output_filepath)
    
    assert(np.any(X == 0) == False)    # Make sure 1-based indexing
    assert(np.shape(X) == np.shape(Y))

    num_input_asts = np.shape(X)[0]
    max_timestep = 10
    print("Max ast_id (1-based):", np.amax(X))

    X = X[:, np.arange(max_timestep)]
    Y = Y[:, np.arange(max_timestep)]

    split_ind = int((1 - 0.1) * num_input_asts)
    X_train = X[np.arange(split_ind), :]
    Y_train = Y[np.arange(split_ind), :]
    X_validate = X[np.arange(split_ind, num_input_asts), :]
    Y_validate = Y[np.arange(split_ind, num_input_asts), :]

    data = {"X_train": X_train, "Y_train": Y_train,
            "X_validate": X_validate, "Y_validate": Y_validate}

    print("Fraction of ones in Y_train:", np.mean(Y_train))
    print("Fraction of ones in Y_validate:", np.mean(Y_validate))

    embed_mat = embeddings.load_embeddings(embed_dict_filename)
    
    embed_dims = 100    # ignored if embedding matrix is fed in
    pred_model = predictions.create_baseline_model(X, embed_dims)
    predictions.fit_model(pred_model, data, epochs=1)

    pred_model = predictions.create_nn_model(X, embed_dims)
    predictions.fit_model(pred_model, data, epochs=1) 

    #TODO: you can use this to plot recalls for each timestep.
    #predictions.plot_timestep_recall(model, data, Y_validate)
    

if __name__=="__main__":
    main()
