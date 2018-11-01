from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import mnist

import numpy as np
# from utils import *

class Dataset:
    def __init__(self, args):
        self.mnist = input_data.read_data_sets("./", one_hot=True)
        self.num_samples = self.mnist.train._num_examples
        self.batch_size = args.batch_size
        self.num_batches = int(self.num_samples / self.batch_size)
        print(self.num_samples)

    def get_test_data(self):
        return (self.mnist.test.images, self.mnist.test.labels)

    def get_train_data(self):
        return self.mnist.train

    def get_validation_data(self):
        return (self.mnist.validation.images, self.mnist.validation.labels)


class MnistKeras:
    def __init__(self):
        print('fetching mnist from Keras...')
        (X_train,y_train),(self.X_test,self.y_test) = mnist.load_data()
        print('finished.')
        
        


def check_obj(obj_str):
    obj = eval(obj_str)
    obj_type = type(obj)
    print(obj_str,obj_type,end=' ')
    if obj_type == np.ndarray:
        print(obj.shape)
    else:
        try:
            iterator = iter(obj)
        except TypeError:
            # not iterable
            print()
        else:
            # iterable
            print(len(obj))

if __name__ == '__main__':
    # mnist_keras = MnistKeras()

    # test tensorflow
    mnist = input_data.read_data_sets("./",one_hot=True)
    n_samples = mnist.train._num_examples
    batch_size = 32
    X_train = mnist.train.images
    y_train = mnist.train.labels
    X_train_batch,y_train_batch = mnist.train.next_batch(batch_size)
    X_test = mnist.test.images
    y_test = mnist.test.labels
    check_obj('X_train_batch')
    check_obj('y_train_batch')
    




