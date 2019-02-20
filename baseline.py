#/usr/bin/python3
import numpy as np
import utils

def random_prediction(input_matrix):
    """ Returns prediction matrix (one-hot encoded). """
    num_asts, num_timesteps, num_blocks = np.shape(input_matrix)
    
    decoded_input = np.zeros((num_asts, num_timesteps))
    decoded_predictions = np.zeros(np.shape(decoded_input))
    predictions = np.zeros(np.shape(input_matrix))

    for ast_id in range(num_asts):
        block_indices = utils.onehot_decode(input_matrix[ast_id,:,:]).T
        end_block_mask = (block_indices == (num_blocks - 1)) 
        
        end_block_fill = (num_blocks - 1) * np.ones((1, num_timesteps))
        random_fill = np.random.randint(num_blocks, size=(1, num_timesteps))

        decoded_predictions[ast_id, :] = \
            end_block_mask * end_block_fill + \
            (end_block_mask == False) * random_fill
        
        predictions[ast_id,:,:] = \
            utils.onehot_encode(decoded_predictions[ast_id, :], num_blocks)

    return predictions



if __name__=="__main__":
    all_matrix = utils.load_data("./data-created/q4_array_of_ast_matrices.npy")
    validation_matrix = utils.create_validation_matrix(all_matrix)
    prediction_matrix = random_prediction(all_matrix)
    accuracy, accuracy_valid = \
        utils.accuracy_from_onehot(prediction_matrix, validation_matrix)
    print(accuracy, accuracy_valid)
