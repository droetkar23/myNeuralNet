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

    def feedforward(self, input_layer):

        if not (len(input_layer[0]) == self.layer_sizes[0]):
            print('The input does not match the neural net.')

        self.activations[0] = input_layer

        for i, a, w, b in zip(range(self.layer_number), self.activations, self.weights, self.biases):
            self.activations[i + 1] = sigmoid(np.dot(a, w) + b)
        return self.activations[-1]

    def backprop(self, output_layer):

        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        z = [np.dot(a, w) + b for a, w, b in zip(self.activations, self.weights, self.biases)]


        nabla_a = output_layer
        print('nabla_a = output_layer')
        print(nabla_a.shape, '=', output_layer.shape)
        # nabla_w[-1] = np.outer(self.activations[-2], sigmoid_derivative(z[-1]) * nabla_a)
        # nabla_b[-1] = sigmoid_derivative(z[-1]) * nabla_a

        for i, a, w, z, n_w, n_b in zip(range(self.layer_number - 1, 0, -1), self.activations[-2::-1],
                                        self.weights[::-1], z[::-1],
                                        nabla_w[::-1], nabla_b[::-1]):

            print(i)

            print('nabla_w = activations x dsigm*nabla_a')
            print(n_w.shape, '= ', a.shape, 'x', z.shape, '*', nabla_a.shape)

            nabla_a = np.inner(sigmoid_derivative(z) * nabla_a, w)
            # print('nabla_a= <w,sigma!(z)*nabla_a(before)')
            # print(nabla_a.shape, '= <', w.shape, ',',  sigmoid_derivative(z).shape, '* nabla_a(before)>')
            # nabla_w[i] = np.outer(a, sigmoid_derivative(z) * nabla_a)
            # nabla_b[i] = sigmoid_derivative(z) * nabla_a


        return nabla_w, nabla_b


nn = NeuralNet(layer_sizes=[4, 3, 2])

input_data = np.array([[i for i in range(4)]])

result = nn.feedforward(input_data)
print('result: ')
print(result)
nn.backprop(np.array([[1, 2]]))
# print('biases: ', nn.biases)
# print('weights: ', nn.weights)
#
# print('zip gives:')
# print(list(zip(range(nn.layer_number), nn.activations, nn.biases, nn.weights)))
