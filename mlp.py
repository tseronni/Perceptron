# Copyright (c) 2023 Thiago Seronni Mendonça
# Licensed under the MIT license. See the LICENSE file in the root directory for details.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MLP:

    def __init__(self,
                 training_sample: np.array,
                 class_samples: np.array,
                 hidden_layer: tuple = (1, 2),
                 learning_rate: float = 0.01,
                 epochs: int = 100):

        self.training_sample = training_sample
        self.class_samples = class_samples
        self.hidden_layer = hidden_layer
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Control variables
        self.converged = False
        self.i_epoch = 0
        self.iterations = 0
        self.number_hidden_layers, self.number_of_neurons = self.hidden_layer

        self.number_of_samples, self.number_of_features = training_sample.shape

        # Randomize weights for hidden layer
        # 3 rows x 2 columns, if hidden layer = 1. rows will be number of inputs + bias and columns will be number hidden layers + 1
        # first column is weights between inputs and hidden layer
        # second column is weights between hidden layer and output layer
        # the +1 is for bias
        self.hidden_weights = np.random.rand(self.number_of_features + 1, self.number_hidden_layers + 1) - 0.5

        # Randomize weights for output layer
        # The number of weights is the number of neurons at last hidden layer + 1 bias
        self.output_weights = np.random.rand(self.number_of_neurons + 1) - 0.5

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    # def sigmoid_derivative(self, z):
    #     return self.sigmoid(z) * (1 - self.sigmoid(z))

    @staticmethod
    def sigmoid_derivative(a):
        return a * (1 - a)

    def training(self):
        iteration = 0

        while (self.i_epoch < self.epochs) or (self.converged is True):

            errors = 0
            self.i_epoch += 1

            for inputs, label in zip(self.training_sample, self.class_samples):

                # FEEDFORWARD
                a = np.ones((self.number_hidden_layers, self.number_of_features + 1))
                a[0] = np.insert(inputs, 0, 1)

                derivative = np.ones((self.number_hidden_layers, self.number_of_features))

                for layer_number in range(0, self.number_hidden_layers):
                    z_array = self.hidden_weights.T.dot(a[layer_number])
                    a_array = self.sigmoid(z_array)
                    a = np.vstack((a, np.insert(a_array, 0, 1)))

                    a_derivative = self.sigmoid_derivative(a_array)
                    derivative = np.vstack((derivative, a_derivative))

                # calculate output weights
                last_a = a[-1]
                last_derivative = derivative[-1]
                z_output = self.output_weights.T.dot(last_a)
                a_output = self.sigmoid(z_output)

                y_predict = a_output

                # Error
                loss_function = label - y_predict

                # Delta output = loss_function * ActivationFunctionDerivative
                # Delta is a measure of error in a specific layer (in output layer in this case)
                # You can see Delta as the number that gives the magnitude to the gradient vector
                # The Activation Derivative function is how sensitive a neuron's output is relative to weights
                delta_output_layer = loss_function * self.sigmoid_derivative(y_predict)

                # Gradient will be a vector that indicates the direction and magnitude to maximize the erro
                # This is why we decrease the gradient when we are updating the weights
                # the bias gradient is ignored because in bias the gradient will be de delta itself
                gradient = delta_output_layer * last_a[1:]

                # Update output weights
                self.output_weights[1:] -= gradient * self.learning_rate
                self.output_weights[0] -= delta_output_layer * self.learning_rate

                # BACKPROPAGATION
                # Delta hidden = Weight * DeltaOutput * ActivationFunctionDerivative
                deltas = [delta_output_layer]

                delta_first_reverse_layer = self.hidden_weights * delta_output_layer * derivative[-1]
                self.hidden_weights += delta_first_reverse_layer * self.learning_rate

                # for layer_number in range(self.number_hidden_layers, 0, -1):
                #     delta_hidden = self.hidden_weights[1:, layer_number] * delta_output_layer * derivative[layer_number]
                #     deltas.insert(0, delta_hidden)
                #
                #     gradient_hidden = delta_hidden * a[layer_number - 1]
                #
                #     self.hidden_weights[1:, layer_number] -= gradient_hidden * self.learning_rate
                #     self.hidden_weights[0, layer_number] -= delta_hidden * self.learning_rate

                errors += loss_function ** 2

                # Check convergence
            mse = errors / self.number_of_samples
            if mse < 1e-5:
                self.converged = True

            iteration += 1
            if iteration % 10 == 0:
                print(f'Epoch {self.i_epoch}, MSE: {mse}')

        print('Training completed.')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    input_file_name = 'and.csv'

    df = pd.read_csv(input_file_name)

    training = df.iloc[:, 0:-1].values
    labels = df.iloc[:, -1].values

    model = MLP(training_sample=training, class_samples=labels, learning_rate=0.1, epochs=100)
    model.training()
