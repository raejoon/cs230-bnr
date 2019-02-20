import unittest
import utils
import numpy as np

class TestOneHotEncoding(unittest.TestCase):
    
    def test_encoding(self):
        array = [1, 4]
        num_classes = 5
        one_hot_matrix = utils.onehot_encode(array, num_classes)
        answer = np.array([[0, 1, 0, 0, 0], [0, 0, 0, 0, 1]])
        self.assertTrue(np.array_equal(one_hot_matrix, answer))
        

    def test_decoding(self):
        one_hot_matrix = np.array([[0, 1, 0, 0, 0], [0, 0, 0, 0, 1]])
        class_id_vector = utils.onehot_decode(one_hot_matrix)
        answer = np.reshape(np.array([1, 4]), (2, 1))
        self.assertTrue(np.array_equal(class_id_vector, answer))

if __name__ == "__main__":
    unittest.main()
