# myNeuralNet

A little program containing a class that creates multi-layer-perceptrons that learn to categorize data by optimizing weights and biases via stochastic gradient descent using a backpropagation algorithm.

The 'data/mnist_data' folder contains the mnist handwritten digits data set in the original idx format.

Some pre trained networks can be found in the 'saved_NeuralNets' folder.


needed packages: numpy, idx2numpy (https://github.com/ivanyu/idx2numpy)


usage:

run the 'mnist_loader.py' script to load the mnist data set

to use the use the NeuralNet class:

from neural_net import NeuralNet

use the classes methods to create/load/save a network, train it, evaluate it or show a random example of a guess.

