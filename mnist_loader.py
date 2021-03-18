import idx2numpy
import numpy as np

# import images from the idx files as an ndarray of (28,28) ndarrays of type uint8
mnist_training_images = idx2numpy.convert_from_file('data/mnist_data/train-images.idx3-ubyte')
mnist_test_images = idx2numpy.convert_from_file('data/mnist_data/t10k-images.idx3-ubyte')

# reshape the image arrays, normalize brightness range from (0,255) to (0,1), convert to list
mnist_training_images = [np.reshape(img,(1,784))/255 for img in mnist_training_images]
mnist_test_images = [np.reshape(img,(1,784))/255 for img in mnist_test_images]


# import the labels as an ndarray of ints with range 0 to 9
mnist_training_labels = idx2numpy.convert_from_file('data/mnist_data/train-labels.idx1-ubyte')
mnist_test_labels = idx2numpy.convert_from_file('data/mnist_data/t10k-labels.idx1-ubyte')

#function to reformat the labels as output vectors
def vectorized_result(j):
    e = np.zeros(10)
    e[j] = 1.0
    e = e.reshape(1,10)
    return e

mnist_training_labels = [vectorized_result(label) for label in mnist_training_labels]
mnist_test_labels = [vectorized_result(label) for label in mnist_test_labels]

mnist_training_data = [(ti, to) for ti, to in zip(mnist_training_images, mnist_training_labels)]
mnist_test_data = [(ti, to) for ti, to in zip(mnist_test_images, mnist_test_labels)]