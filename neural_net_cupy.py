
import cupy as cp


def sigmoid(z):
    return 1.0 / (1.0 + cp.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * sigmoid(-z)


class NeuralNet:

    def __init__(self, layer_sizes=(1, 1, 1)) -> None:
        cp.random.seed(1)
        self.layer_sizes = layer_sizes
        self.layer_number = len(layer_sizes)
        self.activations = [cp.zeros((1, y)) for y in layer_sizes]
        # biases start at second layer, first entry belongs to second layer etc.
        self.biases = [cp.random.randn(1, y) for y in layer_sizes[1:-1]]
        self.biases.append(cp.zeros(self.activations[-1].shape))
        # weights start at second layer, first entry belongs to second layer etc.
        self.weights = [cp.random.randn(y, x) for x, y in zip(layer_sizes[1:], layer_sizes[:-1])]


    def think(self, input_layer):

        self.activations[0] = cp.array(input_layer)

        for i, a, w, b in zip(range(self.layer_number), self.activations, self.weights, self.biases):
            self.activations[i + 1] = sigmoid(cp.dot(a, w) + b)

        return self.activations[-1]


    def gradient(self, input, expected_output):

        self.think(input)   # calculate the activations from 'input' that are needed for the backpropagation algorithm

        nabla_w = [cp.zeros(w.shape) for w in self.weights] # initialise the matrices for the w_gradient
        nabla_b = [cp.zeros(b.shape) for b in self.biases]  # initialise the matrices for the b_gradient
        z = [cp.dot(a, w) + b for a, w, b in zip(self.activations, self.weights, self.biases)] # a = sigmoid(z) ...

        nabla_a = self.activations[-1] - cp.array(expected_output)

        for i, a, z, w in zip(range(self.layer_number-2, -1, -1),
                              self.activations[-2::-1], z[::-1], self.weights[::-1]):

            nabla_w[i] = cp.outer(a, nabla_a * sigmoid_derivative(z))
            nabla_b[i] = nabla_a * sigmoid_derivative(z)
            nabla_a = cp.inner(nabla_a * sigmoid_derivative(z), w)

        nabla_b[-1] = cp.zeros_like(nabla_b[-1])

        return (nabla_w, nabla_b)

    def make_step(self, batch, eta):

        nabla_w = [cp.zeros_like(w) for w in self.weights]
        nabla_b = [cp.zeros_like(b) for b in self.biases]

        for i, o in batch:
            delta_nabla_w, delta_nabla_b = self.gradient(i,o)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

        self.weights = [w - (eta/len(batch[0])) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(batch[0])) * nb for b, nb in zip(self.biases, nabla_b)]



    def learn(self, learning_rate: float = 0.1, epochs: int = 1, training_inputs=None, training_outputs=None):

        for epoch in range(epochs):
            pass

