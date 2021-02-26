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
        self.biases = [np.random.randn(1, y) for y in layer_sizes[1:]]
        # weights start at second layer, first entry belongs to second layer etc.
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[1:], layer_sizes[:-1])]

    # def manual_set(self):
    #     for index, item in enumerate(self.layer_sizes):
    #         for a in range(self.layer_sizes[index]):
    #             input('Provide')

    def think(self, input_layer):

        if not (len(input_layer[0]) == self.layer_sizes[0]):
            print('The input does not match the neural net.')

        self.activations[0] = input_layer

        for i, a, w, b in zip(range(self.layer_number), self.activations, self.weights, self.biases):
            self.activations[i + 1] = sigmoid(np.dot(a, w) + b)
        return self.activations[-1]

    def gradient(self):

        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        z = [np.dot(a, w) + b for a, w, b in zip(self.activations, self.weights, self.biases)]

        nabla_a = self.activations[-1]
        print('nabla_a = output_layer')
        print(nabla_a.shape, '=', self.activations[-1].shape)

        for a, w, z, n_w, n_b in zip(self.activations[-2::-1], self.weights[::-1], z[::-1],
                                     nabla_w[::-1], nabla_b[::-1]):

            nabla_w = np.outer(a, sigmoid_derivative(z) * nabla_a)
            nabla_b = sigmoid_derivative(z) * nabla_a
            nabla_a = np.inner(sigmoid_derivative(z) * nabla_a, w)

            # print('nabla_w = activations x dsigm*nabla_a')
            # print(n_w.shape, '= ', a.shape, 'x', z.shape, '*', nabla_a.shape)
            # print('nabla_a= <w,sigma!(z)*nabla_a(before)')
            # print(nabla_a.shape, '= <', w.shape, ',',  sigmoid_derivative(z).shape, '* nabla_a(before)>')

        return nabla_w, nabla_b

    def learn(self, learning_rate: float = 0.1, epochs: int = 1):
        for step in range(epochs):
            self.weights += - learning_rate * self.gradient()[0]
            self.biases += - learning_rate * self.gradient()[1]


nn = NeuralNet(layer_sizes=[5, 4, 3, 2])

input_data = np.array([[i for i in range(5)]])

result = nn.think(input_data)
print('result: ')
print(result)
print(nn.gradient())

# print('biases: ', nn.biases)
# print('weights: ', nn.weights)
#
# print('zip gives:')
# print(list(zip(range(nn.layer_number), nn.activations, nn.biases, nn.weights)))
