import unittest
import os, shutil
import numpy as np
from keras.models import Model, Sequential
import predictions, embeddings

class TestEmbeddings(unittest.TestCase):
    def test_load_asts_from_file(self):
        X1 = np.zeros((2, 2, 4))
        X1[0,:,:] = np.array([[1, 2, 3, 4], [2, 3, 4, 5]])
        X1[1,:,:] = np.array([[10, 20, 30, 40], [20, 30, 40 ,50]])
        embeddings.save_asts_to_file(X1, "tmp.npy")
        X2 = embeddings.load_asts_from_file("tmp.npy")
        np.testing.assert_equal(X1, X2)

    def test_get_output_labels(self):
        X = np.zeros((2, 4, 3))
        X[0,:,:] = np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]])
        X[1,:,:] = np.array([[1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1]])
        Y = embeddings.get_output_labels(X)

        ref = np.zeros((2, 4, 3))
        ref[0,:,:] = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]])
        ref[1,:,:] = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]) 
        np.testing.assert_equal(Y, ref)

    def test_create_model(self):
        X = np.zeros((2, 4, 3))
        X[0,:,:] = np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]])
        X[1,:,:] = np.array([[1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1]])
        self.assertEqual(type(embeddings.create_model(X)), type(Sequential()))
        

class TestPredictions(unittest.TestCase):
    def test_load_trajectories_from_dataset(self):
        os.makedirs("tmp_traj")
        try:
            with open("tmp_traj/idMap.txt", "w") as fo:
                fo.write("secretId,trajectoryId\n")
                fo.write("0,1\n1,2\n2,2")
            with open("tmp_traj/1.txt", "w") as fo:
                fo.write("10\n20")
            with open("tmp_traj/2.txt", "w") as fo:
                fo.write("200\n400\n600\n800")
            X = predictions.load_trajectories_from_dataset("tmp_traj")
            reference = np.array([[10, 20, 20, 20],
                                  [200, 400, 600, 800],
                                  [200, 400, 600, 800]])
            np.testing.assert_equal(X, reference)
        except:
            shutil.rmtree("tmp_traj")
            raise
        
        shutil.rmtree("tmp_traj")

    def test_load_trajectories_from_file(self):
        X1 = np.array([[10, 20, 20, 20],
                      [200, 400, 600, 800],
                      [200, 400, 600, 800]])
        predictions.save_trajectories_to_file(X1, "tmp.npy")
        X2 = predictions.load_trajectories_from_file("tmp.npy")
        os.remove("tmp.npy")
        np.testing.assert_equal(X1, X2)

    def test_output_labels_csv(self):
        filename = "tmp.csv"
        Y1 = np.array([[0, 1, 1, 0], [1, 0, 1, 0]])
        predictions.save_output_labels_csv(Y1, filename)
        Y2 = predictions.load_output_labels_csv(filename)
        os.remove(filename)
        np.testing.assert_equal(Y1, Y2)

    def test_create_nn_model_without_embeddings(self):
        X = np.array([[0, 1, 2, 3], [3, 2, 1, 0]])
        model = predictions.create_nn_model(X, 4, 2) 
        self.assertEqual(type(model), type(Sequential()))

    def test_create_nn_model_with_embeddings(self):
        # Create model
        X = np.array([[0, 1, 2, 3], [3, 2, 1, 0]])
        embeddings_matrix = \
            np.array([[0., 0.], [0.1, 0.2], [0.2, 0.3], [0.3, 0.2], [0.2, 0.1]])
        model = predictions.create_nn_model(X, 4, 2, embeddings_matrix) 
        self.assertEqual(type(model), type(Sequential()))
        
        # Get embedding layer output and make sure it is correct.
        embedding_layer = Model(inputs=model.input,
                                outputs=model.layers[0].output)
        embedding_output = embedding_layer.predict(X)
        
        reference = np.zeros((np.shape(X)[0], np.shape(X)[1],
                              np.shape(embeddings_matrix)[1]))
        reference[0,:,:] = np.array([[0., 0.], [.1, .2], [.2, .3], [.3, .2]])
        reference[1,:,:] = np.array([[.3, .2], [.2, .3], [.1, .2], [0., 0.]])
        np.testing.assert_almost_equal(reference, embedding_output)

    def test_create_baseline_model_without_embeddings(self):
        X = np.array([[0, 1, 2, 3], [3, 2, 1, 0]])
        model = predictions.create_baseline_model(X, 4, 2) 
        self.assertEqual(type(model), type(Sequential()))

    def test_create_baseline_model_with_embeddings(self):
        # Create model
        X = np.array([[0, 1, 2, 3], [3, 2, 1, 0]])
        embeddings_matrix = \
            np.array([[0., 0.], [0.1, 0.2], [0.2, 0.3], [0.3, 0.2], [0.2, 0.1]])
        model = predictions.create_baseline_model(X, 4, 2, embeddings_matrix) 
        self.assertEqual(type(model), type(Sequential()))
        
        # Get embedding layer output and make sure it is correct.
        embedding_layer = Model(inputs=model.input,
                                outputs=model.layers[0].output)
        embedding_output = embedding_layer.predict(X)
        
        reference = np.zeros((np.shape(X)[0], np.shape(X)[1],
                              np.shape(embeddings_matrix)[1]))
        reference[0,:,:] = np.array([[0., 0.], [.1, .2], [.2, .3], [.3, .2]])
        reference[1,:,:] = np.array([[.3, .2], [.2, .3], [.1, .2], [0., 0.]])
        np.testing.assert_almost_equal(reference, embedding_output)
        

if __name__=="__main__":
    unittest.main()
