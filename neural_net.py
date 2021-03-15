
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
        self.activations = [np.zeros(y) for y in layer_sizes]
        # biases start at second layer, first entry belongs to second layer etc.
        self.biases = [np.random.randn(1, y) for y in layer_sizes[1:-1:]]
        self.biases.append(np.zeros(self.activations[-1].shape))
        # weights start at second layer, first entry belongs to second layer etc.
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[1:], layer_sizes[:-1])]

    # def manual_set(self):
    #     for index, item in enumerate(self.layer_sizes):
    #         for a in range(self.layer_sizes[index]):
    #             input('Provide')

    def think(self, input_layer):

        # if not (len(input_layer[0]) == self.layer_sizes[0]):
        #     print('The input does not match the neural net.')

        self.activations[0] = np.array(input_layer)

        for i, a, w, b in zip(range(self.layer_number), self.activations, self.weights, self.biases):
            print(i)
            self.activations[i + 1] = sigmoid(np.dot(a, w) + b)

        self.activations[-1] = sigmoid(np.dot(self.activations[-2], self.weights[-1]))

        return self.activations[-1]

    def gradient(self, training_example):

        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]  # problem if empty then we cant use shape later on
        z = [np.dot(a, w) + b for a, w, b in zip(self.activations, self.weights, self.biases)]

        nabla_a = self.activations[-1]-np.array(training_example)
        print(nabla_a.shape)

        for i, a, z, w in zip(range(self.layer_number-2, -1, -1),
                              self.activations[-2::-1], z[::-1], self.weights[::-1]):

            print(i)
            print(nabla_a.shape)
            nabla_w[i] = np.outer(a*sigmoid_derivative(z), nabla_a)
            #nabla_b[i] = sigmoid_derivative(z)*nabla_a
            nabla_a = np.inner(nabla_a*sigmoid_derivative(z), w)


        return nabla_w  # , nabla_b

    def learn(self, learning_rate: float = 0.1, epochs: int = 1, training_inputs=None, training_outputs=None):

        for step in range(epochs):

            nabla_w = np.empty(shape=(self.weights,training_inputs.shape[0]))
            for ex_num, i,o in zip(range(3),training_inputs, training_outputs):
                print(ex_num)
                self.think(i)
                nabla_w[ex_num]=np.array(self.gradient(o))
                #nabla_w, nabla_b = self.gradient(o)

            nabla_w.mean(axis=1)
            self.weights = [w - learning_rate * n_w for w, n_w in zip(self.weights, nabla_w)]
            #self.biases = [b - learning_rate * nabla_b for b, nabla_b in zip(self.biases,nabla_b)]


training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

training_outputs = np.array([[0],
                             [1],
                             [1],
                             [0]])

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
