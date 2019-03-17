from keras.models import load_model
import os

def save_model(model, filepath):    
    model.save(filepath)
    
    # Serialize model to JSON
    model_json = model.to_json()
    archpath = filepath + ".json"
    with open(archpath, "w") as json_file:
        json_file.write(model_json)  
        
    print("Saved model to " + filepath )    
    

def load_saved_model(filepath):
    model = load_model(filepath)
    print("Loaded model from " + filepath)
    return model

"""
def get_ast_ids(ast_count_filepath):
    with open(ast_count_filepath) as fo:
        fo.readline()
        existing_ids = [int(line.rstrip().split()[0]) for line in fo]
    return existing_ids
"""

def get_ast_ids(ast_dirpath):
    ast_ids = []
    for subdir, dirs, files in os.walk(ast_dirpath):
        for filename in files:
            front, ext = os.path.splitext(filename)
            if ext.lower() == ".json":
                ast_id = int(os.path.basename(front))
                if ast_id != 51:
                    ast_ids.append(ast_id)
    return ast_ids
