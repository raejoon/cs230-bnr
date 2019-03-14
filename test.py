import unittest
import utils
import numpy as np
from keras.utils import to_categorical

class TestOneHotEncoding(unittest.TestCase):
    
    def test_encoding(self):
        array = np.array([1, 4])
        num_classes = 5
        one_hot_matrix = utils.onehot_encode(array, num_classes)
        answer = np.array([[0, 1, 0, 0, 0], [0, 0, 0, 0, 1]])
        self.assertTrue(np.array_equal(one_hot_matrix, answer))
        

    def test_decoding(self):
        one_hot_matrix = np.array([[0, 1, 0, 0, 0], [0, 0, 0, 0, 1]])
        class_id_vector = utils.onehot_decode(one_hot_matrix)
        answer = np.reshape(np.array([1, 4]), (2, 1))
        self.assertTrue(np.array_equal(class_id_vector, answer))


class TestBlockPredictionAccuracy(unittest.TestCase):
    
    def test_accuracy_from_class(self):
        prediction_matrix = np.array([[2, 1, 0, 3, 3], [1, 3, 3, 3, 3]])
        validation_matrix = np.array([[2, 1, 1, 3, 3], [0, 3, 3, 3, 3]])
        
        accuracy = utils.accuracy_from_class_matrix(prediction_matrix, 
                                                    validation_matrix, 3)
        self.assertEqual(accuracy, 0.5)

    def test_accuracy_from_onehot(self):
        prediction_matrix = np.array([[2, 1, 0, 3, 3], [1, 3, 3, 3, 3]])
        validation_matrix = np.array([[2, 1, 1, 3, 3], [0, 3, 3, 3, 3]])

        prediction_matrix = to_categorical(prediction_matrix, num_classes=4) 
        validation_matrix = to_categorical(validation_matrix, num_classes=4)

        accuracy = utils.accuracy_from_onehot_matrix(prediction_matrix, 
                                                     validation_matrix)
        self.assertEqual(accuracy, 0.5)
        


if __name__ == "__main__":
    unittest.main()
