# myNeuralNet

A little program containing a class that creates multi-layer-perceptrons that learn to categorize data by optimizing weights and biases via stochastic gradient descent using a backpropagation algorithm.

The 'data/mnist_data' folder contains the mnist data set in the original idx format as well as a pickled version of the pre processed data which was created using idx2numpy.

Some pre trained networks can be found in the saved_NeuralNets folder.


needed packages: numpy


usage:

to access the mnist data, use the pickle package: 
mnist_training_data, mnist_test_data = pickle.load(open('data/mnist_data/mnist_data_pickled.p', 'rb'))

to use the use the NeuralNet class:
from neural_net load NeuralNet

