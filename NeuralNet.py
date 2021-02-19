import numpy as np


class NeuralNet:

    def __init__(self, layer_sizes=(1, 1, 1)) -> None:
        np.random.seed(1)
        self.layer_sizes = layer_sizes
        self.layer_number = len(layer_sizes)
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[1:], layer_sizes[:-1])]

    # def manual_set(self):
    #     for index, item in enumerate(self.layer_sizes):
    #         for a in range(self.layer_sizes[index]):
    #             input('Provide')

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def feedforward(self, a):
        # if not (len(a[0]) == self.layer_sizes[0]): print('The input does not match the neural net.')
        for b, w in zip(self.biases, self.weights):
            print(np.shape(w))
            print(np.shape(a))
            print(np.dot(a, w))
            print(b.T)
            a = self.sigmoid(np.dot(a, w) + b.T)
        return a


nn = NeuralNet(layer_sizes=[3, 4, 5, 3, 1])
print('Synaptic weights: ')
print(nn.weights)
print('biases:')
print(nn.biases)

input_data = np.array([[1, 2, 3]])

result = nn.feedforward(input_data)
print('result: ')
print(result)
