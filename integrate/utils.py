from keras.models import load_model

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
