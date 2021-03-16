from typing import List, Any, Tuple

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * sigmoid(-z)


class NeuralNet:

    def __init__(self, layer_sizes=(1, 1, 1)) -> None:
        np.random.seed(1)
        self.layer_sizes = layer_sizes
        self.layer_number = len(layer_sizes)
        self.activations = [np.zeros((1,y)) for y in layer_sizes]
        # biases start at second layer, first entry belongs to second layer etc.
        self.biases = [np.random.randn(1, y) for y in layer_sizes[1:-1]]
        self.biases.append(np.zeros(self.activations[-1].shape))
        # weights start at second layer, first entry belongs to second layer etc.
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[1:], layer_sizes[:-1])]

    # def manual_set(self):
    #     for index, item in enumerate(self.layer_sizes):
    #         for a in range(self.layer_sizes[index]):
    #             input('Provide')

    def think(self, input_layer):

        self.activations[0] = np.array(input_layer)

        for i, a, w, b in zip(range(self.layer_number), self.activations, self.weights, self.biases):
            self.activations[i + 1] = sigmoid(np.dot(a, w) + b)

        return self.activations[-1]


    def gradient(self, input, expected_output):

        self.think(input)   # calculate the activations from 'input' that are needed for the backpropagation algorithm

        nabla_w = [np.zeros(w.shape) for w in self.weights] # initialise the matrices for the w_gradient
        nabla_b = [np.zeros(b.shape) for b in self.biases]  # initialise the matrices for the b_gradient
        z = [np.dot(a, w) + b for a, w, b in zip(self.activations, self.weights, self.biases)] # a = sigmoid(z) ...

        nabla_a = self.activations[-1]-np.array(expected_output)

        for i, a, z, w in zip(range(self.layer_number-2, -1, -1),
                              self.activations[-2::-1], z[::-1], self.weights[::-1]):

            nabla_w[i] = np.outer(a, nabla_a * sigmoid_derivative(z))
            nabla_b[i] = nabla_a * sigmoid_derivative(z)
            nabla_a = np.inner(nabla_a*sigmoid_derivative(z), w)

        nabla_b[-1] = np.zeros_like(nabla_b[-1])

        return (nabla_w, nabla_b)

    def make_step(self, batch, eta):

        nabla_w = [np.zeros_like(w) for w in self.weights]
        nabla_b = [np.zeros_like(b) for b in self.biases]

        for i, o in batch:
            delta_nabla_w, delta_nabla_b = self.gradient(i,o)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

        self.weights = [w - (eta/len(batch[0])) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(batch[0])) * nb for b, nb in zip(self.biases, nabla_b)]



    def learn(self, learning_rate: float = 0.1, epochs: int = 1, training_inputs=None, training_outputs=None):

        for epoch in range(epochs):
            pass





training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

training_outputs = np.array([[0],
                             [1],
                             [1],
                             [0]])

training_data = [(ti,to) for ti, to in zip(training_inputs,training_outputs)]

# result = nn.think(input_data)
# print('result: ')
# print(result)

nn = NeuralNet(layer_sizes=[3,1])
#small_network.think(np.array([[1,1,1]]))
#small_network.learn(learning_rate=3.0, epochs=1, training_inputs=training_inputs, training_outputs=training_outputs)

#print('layer_number', small_network.layer_number)
# print(small_network.weights)
#print('weights=', small_network.weights)
#print('biases= ', small_network.biases)


# print('biases: ', nn.biases)
# print('weights: ', nn.weights)
#
# print('zip gives:')
# print(list(zip(range(nn.layer_number), nn.activations, nn.biases, nn.weights)))
