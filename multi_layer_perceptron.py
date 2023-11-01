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

        self.weights = []

        # Xavier (Glorot) initialization
        for layer_number in range(self.number_hidden_layers):
            self.weights.append(np.random.rand(self.number_of_neurons, self.number_of_features + 1) - 1)

        self.weights.append(np.random.rand(1, self.number_of_features + 1) - 1)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(a):
        return a * (1 - a)

    def training(self):
        iteration = 0

        while self.i_epoch < self.epochs and not self.converged:

            squared_errors = 0
            self.i_epoch += 1

            for inputs, label in zip(self.training_sample, self.class_samples):

                # FEEDFORWARD
                a = np.ones((self.number_hidden_layers, self.number_of_features + 1))
                a[0] = np.insert(inputs, 0, 1)

                derivative = np.ones((self.number_hidden_layers, self.number_of_features + 1))

                for layer_number in range(0, self.number_hidden_layers):
                    z_array = self.weights[layer_number].dot(a[layer_number])
                    a_array = self.sigmoid(z_array)
                    a = np.vstack((a, np.insert(a_array, 0, 1)))

                    a_derivative = self.sigmoid_derivative(a_array)
                    derivative = np.vstack((derivative, np.insert(a_derivative, 0, 1)))

                # calculate output weights
                z_output = self.weights[-1].dot(a[-1])
                a_output = self.sigmoid(z_output)

                y_predict = a_output[0]

                # Error
                loss_function = label - y_predict

                squared_errors += loss_function ** 2

                # Delta output = loss_function * ActivationFunctionDerivative
                # Delta is a measure of error in a specific layer (in output layer in this case)
                # You can see Delta as the number that gives the magnitude to the gradient vector
                # The Activation Derivative function is how sensitive a neuron's output is relative to weights
                delta_output_layer = loss_function * self.sigmoid_derivative(y_predict)

                # Gradient will be a vector that indicates the direction and magnitude to maximize the error
                # This is why we decrease the gradient when we are updating the weights
                # the bias gradient is ignored because in bias the gradient will be de delta itself
                gradient = delta_output_layer * a[-1]

                # Update output weights (The output weights will be the last weights in the weights array)
                # The bias will be the same value of delta_output_layer
                self.weights[-1] -= gradient * self.learning_rate

                # BACKPROPAGATION
                # Delta output = loss_function * ActivationFunctionDerivative
                # Delta hidden before output layer = Weight * DeltaOutput * ActivationFunctionDerivative
                # Delta hidden = Weight * DeltaNextLayer * ActivationFunctionDerivative
                deltas = [delta_output_layer]

                for layer_number in range(self.number_hidden_layers, 0, -1):

                    delta_hidden = self.weights[layer_number] * deltas[-1] * derivative[layer_number]
                    deltas.append([delta_hidden])

                    gradient_hidden = delta_hidden * a[layer_number - 1]

                    self.weights[layer_number - 1] -= gradient_hidden * self.learning_rate

            mse = squared_errors / self.number_of_samples
            if mse < 1e-5:
                self.converged = True

            iteration += 1
            if iteration % 10 == 0:
                print(f'Epoch {self.i_epoch}, MSE: {mse}')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    input_file_name = 'and.csv'

    df = pd.read_csv(input_file_name)

    training = df.iloc[:, 0:-1].values
    labels = df.iloc[:, -1].values

    model = MLP(training_sample=training, class_samples=labels, learning_rate=0.001, epochs=1000)
    model.training()
