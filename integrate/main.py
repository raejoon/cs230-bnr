import embeddings
import predictions
import utils
import numpy as np

def embeddings_run():
    ast_filepath = "../data-created/q4_array_of_ast_matrices.npy"
    embed_input = embeddings.load_asts_from_file(ast_filepath) 
    embed_output = embeddings.get_output_labels(embed_input)
    
    embed_model = embeddings.create_model(embed_input)
    embed_history = embeddings.fit_model(embed_model, 
                                         embed_input, embed_output,
                                         epochs=10)
    print(embed_history.effective_accuracy["train"])
    print(embed_history.effective_accuracy["validate"])

    embed_model_filename = "tmp/my_embeddings.h5"
    utils.save_model(embed_model, embed_model_filename)
    embed_model = utils.load_saved_model(embed_model_filename)
    
    embed_dict = embeddings.get_embeddings(embed_model, embed_input)
    embed_dict_filename = "tmp/my_embeddings.npy"
    embeddings.save_embeddings(embed_dict, embed_dict_filename)    


def main():
    # Embeddings generation
    embeddings_run()
     
    # Failure prediction (Main task)
    pred_input = predictions.load_trajectories_from_dataset("hoc4/trajectories")
    pred_output = predictions.get_output_labels(pred_output)
    
    embed_dict = embeddings.load_embeddings(embed_dict_filename)
    pred_model = predictions.create_nn_model(pred_input, embed_dict)
    pred_history = predictions.fit_model(pred_model, pred_input, pred_output) 
    
    pred_model_filename = "tmp/my_predictions.h5"
    utils.save_model(pred_model, pred_model_filename)
    
    # Failure prediction with logistic regression 
    pred_model = predictions.create_baseline_model(pred_input)
    pred_history = predictions.fit_model(pred_model, pred_input, pred_output)
    
    pred_model_filename = "tmp/baseline_predictions.h5"
    utils.save_model(pred_model, pred_model_filename)


if __name__=="__main__":
    embeddings_run()
