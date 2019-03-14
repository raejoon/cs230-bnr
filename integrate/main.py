import embeddings
import predictions
import utils

def main():
    # Embeddings generation
    #embed_input = embeddings.load_asts_from_dataset("hoc4/asts")
    embed_input = embeddings.load_asts_from_file("q4_asts.npy") 
    embed_output = embeddings.get_output_labels(pred_input)
    
    embed_model = embeddings.create_model(embed_input)
    embed_history = embeddings.fit_model(embed_model, embed_input, embed_output)

    embed_model_filename = "tmp/my_embeddings.h5"
    utils.save_model(embed_model, embed_model_filename)
    embed_model = utils.load_model(embed_model_filename)
    
    embed_dict = embeddings.get_embeddings(embed_model, embed_input)
    embed_dict_filename = "tmp/my_embeddings.dat"
    embeddings.save_embeddings(embed_dict, embed_dict_filename) 
     
    # Failure prediction (Main task)
    pred_input = predictions.load_trajectories_from_dataset("hoc4/trajectories")
    pred_output = predictions.get_output_labels(pred_output)
    
    embed_dict = embeddings.load_embeddings(embed_dict_filename)
    pred_model = predictions.create_nn_model(pred_input)
    pred_history = predictions.fit_model(pred_model, pred_input, pred_output) 
    
    pred_model_filename = "tmp/my_predictions.h5"
    utils.save_model(pred_model, pred_model_filename)
    
    # Failure prediction with logistic regression 
    pred_model = predictions.create_baseline_model(pred_input)
    pred_history = predictions.fit_model(pred_model, pred_input, pred_output)
    
    pred_model_filename = "tmp/baseline_predictions.h5"
    utils.save_model(pred_model, pred_model_filename)
