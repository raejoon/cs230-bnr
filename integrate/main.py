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
    
    #embed_model = embeddings.create_model(embed_input)
    #embed_history = embeddings.fit_model(embed_model, 
    #                                     embed_input, embed_output,
    #                                     epochs=50)
    #print(embed_history.effective_accuracy["train"])
    #print(embed_history.effective_accuracy["validate"])

    embed_model_filename = "tmp/my_embeddings.h5"
    #utils.save_model(embed_model, embed_model_filename)
    embed_model = utils.load_saved_model(embed_model_filename)
    
    ast_dirpath = "anonymizeddata/data/hoc4/asts/"
    embed_dict = embeddings.get_embeddings(embed_model, embed_input,
                                           ast_dirpath)
    embeddings.save_embeddings(embed_dict, embed_dict_filename)    


def main():
    # Embeddings generation
    embeddings_run()
     
    # Failure prediction (Main task)
    #traj_dirpath = "anonymizeddata/data/hoc4/trajectories"
    #pred_input = predictions.load_trajectories_from_dataset(traj_dirpath)

    input_filepath = "processed/traj_ast_matrix.npy"
    pred_input = predictions.load_trajectories_from_file(input_filepath) + 1

    output_filepath = "processed/traj_fail_window_2_matrix.npy"
    pred_output = predictions.load_output_labels_npy(output_filepath)
    
    #indices = np.random.permutation(np.shape(pred_input)[0])[:100]
    #pred_input = pred_input[indices, :]
    #pred_output = pred_output[indices, :]
    
    pred_input = pred_input[:, np.arange(100)]
    pred_output = pred_output[:, np.arange(100)]

    print(np.mean(pred_output.flatten()))
    
    #embed_mat = embeddings.load_embeddings(embed_dict_filename)
    #embed_dims = np.shape(embed_mat)[1]

    #pred_model = predictions.create_nn_model(pred_input, embed_dims)
    ##pred_model = predictions.create_nn_model(pred_input, embed_dims, embed_mat)
    #pred_history = predictions.fit_model(pred_model, pred_input, pred_output,
    #                                     epochs=5) 

    #predictions.plot_recall_curves(pred_history)
    #plt.savefig("recall_curve.png")
    #pred_model_filename = "tmp/my_predictions.h5"
    #utils.save_model(pred_model, pred_model_filename)
    
    # Failure prediction with logistic regression 
    embed_dims = np.amax(pred_input)
    pred_model = predictions.create_baseline_model(pred_input, 100)
    pred_history, X_train, Y_train, X_validate, Y_validate = \
        predictions.fit_model(pred_model, pred_input, pred_output, epochs=3)

    pred_model_filename = "tmp/baseline_predictions.h5"
    utils.save_model(pred_model, pred_model_filename)

    predictions.plot_timestep_recall(pred_model, X_train, Y_train, X_validate,
            Y_validate)
    

if __name__=="__main__":
    main()
