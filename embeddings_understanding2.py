# Get ready
import utils
from embeddings import *

# Load data
all_matrix = utils.load_data("./data-created/q4_array_of_ast_matrices.npy")
validation_matrix = utils.create_validation_matrix(all_matrix)
print(all_matrix.shape)
print(validation_matrix.shape)

# Split into training/dev/test set
train_matrix, dev_matrix, test_matrix = utils.split_data(all_matrix)
print(train_matrix.shape)
print(dev_matrix.shape)
print(test_matrix.shape)

# Create model
num_timesteps = all_matrix.shape[1]
num_blocks = all_matrix.shape[2]
model = create_model(num_timesteps, num_blocks)

# Train model
train_model(model, all_matrix, validation_matrix)