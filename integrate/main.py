import embeddings
import predictions
import utils
import numpy as np

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
    traj_dirpath = "anonymizeddata/data/hoc4/trajectories"
    pred_input = predictions.load_trajectories_from_dataset(traj_dirpath)

    input_filepath = "processed/hoc4_traj.npy"
    pred_input = predictions.load_trajectories_from_file(input_filepath)

    pred_output = \
        predictions.load_output_labels_npy("processed/correct_within_1_try2.npy")
    
    embed_mat = embeddings.load_embeddings(embed_dict_filename)
    embed_dims = np.shape(embed_mat)[1]
    pred_model = predictions.create_nn_model(pred_input, embed_dims, embed_mat)
    pred_history = predictions.fit_model(pred_model, pred_input, pred_output,
                                         epochs=10) 
    
    pred_model_filename = "tmp/my_predictions.h5"
    utils.save_model(pred_model, pred_model_filename)
    
    # Failure prediction with logistic regression 
    pred_model = predictions.create_baseline_model(pred_input, embed_dims,
                                                   embed_mat)
    pred_history = predictions.fit_model(pred_model, pred_input, pred_output,
                                         epochs=10)
    
    pred_model_filename = "tmp/baseline_predictions.h5"
    utils.save_model(pred_model, pred_model_filename)


if __name__=="__main__":
    main()
