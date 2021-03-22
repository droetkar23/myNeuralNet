import numpy as np
import random
import pickle
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * sigmoid(-z)


class NeuralNet:

    def __init__(self, layer_sizes=(1, 1, 1)) -> None:
        np.random.seed(1)
        self.layer_sizes = layer_sizes
        self.layer_number = len(layer_sizes)
        self.activations = [np.zeros((1, y)) for y in layer_sizes]
        # biases start at second layer, first entry belongs to second layer etc.
        self.biases = [np.random.randn(1, y) for y in layer_sizes[1:-1]]
        self.biases.append(np.zeros(self.activations[-1].shape))
        # weights start at second layer, first entry belongs to second layer etc.
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[1:], layer_sizes[:-1])]

    def save_to_file(self, file_path=''):
        if file_path == '':
            print('Provide a file to save to')
        else:
            pickle.dump(self, open(str(file_path), 'wb'))

    @classmethod
    def load_from_file(cls, file_path=''):
        new_instance = None
        if file_path == '':
            print('Provide a file to load from')
        else:
            new_instance = pickle.load(open(file_path, 'rb'))
        return new_instance

    def think(self, input_layer):

        self.activations[0] = np.array(input_layer)

        for i, a, w, b in zip(range(self.layer_number), self.activations, self.weights, self.biases):
            self.activations[i + 1] = sigmoid(np.dot(a, w) + b)

        return self.activations[-1]

    def gradient(self, input_layer, expected_output):

        self.think(input_layer)  # calculate the activations from 'input' needed for the backpropagation algorithm

        nabla_w = [np.zeros(w.shape) for w in self.weights]  # initialise the matrices for the w_gradient
        nabla_b = [np.zeros(b.shape) for b in self.biases]  # initialise the matrices for the b_gradient
        z = [np.dot(a, w) + b for a, w, b in zip(self.activations, self.weights, self.biases)]  # a = sigmoid(z) ...

        nabla_a = self.activations[-1] - np.array(expected_output)

        for i, a, z, w in zip(range(self.layer_number - 2, -1, -1),
                              self.activations[-2::-1], z[::-1], self.weights[::-1]):
            nabla_w[i] = np.outer(a, nabla_a * sigmoid_derivative(z))
            nabla_b[i] = nabla_a * sigmoid_derivative(z)
            nabla_a = np.inner(nabla_a * sigmoid_derivative(z), w)

        nabla_b[-1] = np.zeros_like(nabla_b[-1])

        return nabla_w, nabla_b

    def make_step(self, batch, eta=1.0):

        nabla_w = [np.zeros_like(w) for w in self.weights]
        nabla_b = [np.zeros_like(b) for b in self.biases]

        for i, o in batch:
            delta_nabla_w, delta_nabla_b = self.gradient(i, o)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

        self.weights = [w - (eta / len(batch[0])) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(batch[0])) * nb for b, nb in zip(self.biases, nabla_b)]

    def learn(self, learning_rate=1.0, epochs=1, training_data=None, test_data=None, batch_size=1, info=None):

        for epoch in range(epochs):

            if info: print('epoch:', epoch + 1)

            random.shuffle(training_data)

            for k in range(0, len(training_data), batch_size):

                current_batch = training_data[k:k + batch_size]

                self.make_step(batch=current_batch, eta=learning_rate)


            if info:
                print('accuracy on training set:', self.evaluate_accuracy(data_set=training_data))

            if info and test_data:
                print('accuracy on test set:', self.evaluate_accuracy(data_set=test_data))

    def evaluate_accuracy(self, data_set):

        correct_classifications = 0

        for example in data_set:

            if example[1].argmax() == self.think(example[0]).argmax():
                correct_classifications += 1

        p = correct_classifications / len(data_set)

        return p
    
    def show_example(self, data_set=None, guessed_correctly=True):

        example_number=0

        while not guessed_correctly:
            example_number = np.random.randint(0,len(data_set))
            if self.think(data_set[example_number][0]).argmax() != data_set[example_number][1].argmax():
                guessed_correctly=True

        img = data_set[example_number][0].reshape(28,28)
        description = 'The nets guess:' + str(self.think(data_set[example_number][0]).argmax()) +\
                        ',the correct label is:' + str(data_set[example_number][1].argmax())

        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        ax.set_title(description)
        plt.show()

