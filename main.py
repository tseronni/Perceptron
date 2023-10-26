import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Perceptron:

    def __init__(self,
                 training_sample: np.array,
                 class_samples: np.array,
                 learning_rate: float = 0.01,
                 epochs: int = 100):

        self.training_sample = training_sample
        self.class_samples = class_samples
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.converged = False
        self.i_epoch = 0
        self.iterations = 0

        self.training_shape = training_sample.shape
        self.number_of_samples = self.training_shape[0]         # Number of rows is the number of samples
        self.number_of_features = self.training_shape[1]        # Number of columns is the number of features

        # Number of weights is the number of features + 1 for bias
        # Initialize weights and bias with random numbers between [-0.5, 0.5]
        # Considering the bias as the first value of the array
        self.weights = np.random.rand(self.number_of_features + 1) - 0.5

    def training(self):

        while (self.i_epoch < self.epochs) or (self.converged is True):

            errors = 0
            self.i_epoch += 1

            for inputs, label in zip(self.training_sample, self.class_samples):
                bias = self.weights[0]
                z = self.weights[1:].T.dot(inputs) + bias

                # For learning purpose we will use the STEP FUNCTION as activation function. (Perceptron was idealized initialize with this function)
                # STEP FUNCTION THRESHOLD
                # if z > 0, a = 1
                # if z <=0, a = 0

                a = 0
                if z > 0:
                    a = 1

                # Because perceptron has only one neuron and this neuron is the output layer, the prediction will be the value of the "a" variable
                y_prediction = a

                loss_function = label - y_prediction

                # OBS1: Perceptron does not exist the concept of Gradient
                # OBS2: Perceptron does not exist the concept of Backpropagation
                # OBS3: Perceptron does not have an explicit Cost Function, because the weights are updated at each sample directly.
                # (Remember that the Cost function is computed after an epoch and loss function is computed at each iteration)
                # OBS4: The Cost Function will be an implicit idea as "The objective is to reduce the loss function"

                self.weights[1:] = self.weights[1:] + (self.learning_rate * loss_function * inputs)     # update weights
                self.weights[0] = self.weights[0] + (self.learning_rate * loss_function)                # update bias

                errors = errors + abs(loss_function)

                self.iterations += 1
                self.plot_decision_boundary(z, inputs[0], inputs[1], label, y_prediction, loss_function)

                d = 0

            # Very important observation!
            # Convergence in the Perceptron is achieved when the model is able to correctly classify all training examples. Ways to do are:

            # 1 - Zero Error at an Epoch: If, during an epoch, the model does not make any classification errors in all training examples, you can consider that it has converged.

            # 2 - Error Reduction: If the number of classification errors decreases at each epoch, it is a sign that the model is learning and moving closer to convergence.
            # You can define a stop criterion based on a sufficiently large drop in the number of errors or a small variance in the error.

            # 3 - Maximum Number of Epochs: You can also set a maximum number of epochs as a stopping criterion. If the maximum number of epochs is reached without
            # convergence being achieved, training may be interrupted

            # In this example I'm using a mix of option 1 and 3.

            if errors == 0:
                self.converged = True
                break

    def plot_decision_boundary(self, z, x1, x2, label, y_pred, loss):
        fig, ax = plt.subplots()

        ax.set_xlim(-1, 2)
        ax.set_ylim(-1, 2)

        for inputs, label in zip(self.training_sample, self.class_samples):
            if label == 0:
                ax.scatter(inputs[0], inputs[1], color='black', marker='o', label='Class 0', s=75)
            else:
                ax.scatter(inputs[0], inputs[1], color='black', marker='^', label='Class 1', s=75)

        # # Highlighting the current point
        circle = plt.Circle((x1, x2), 0.1, color='blue', fill=False, linewidth=2)
        ax.add_artist(circle)

        # Decision Boundary
        x_values = np.linspace(-1, 2, 100)
        y_values = (-self.weights[1] * x_values - self.weights[0]) / self.weights[2]
        ax.plot(x_values, y_values, color='green', linestyle='--')

        # Place the 'X' for y_prediction directly at x1 and x2 with increased size
        ax.scatter(x1, y_pred, color='blue', marker='x', s=300, label='y_prediction')

        ax.set_title(f"Decision Boundary - Iteration {self.iterations} - Epoch {self.i_epoch}")
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")

        v = f'X1={x1}, X2={x2}, label={label}'
        z_formula = f'z = {self.weights[1]:.2f} * X1 + {self.weights[2]:.2f} * X2 + {self.weights[0]:.2f}'
        z_value = f'z = {z:.2f}'
        y_value = f'a = y_prediction = {y_pred}'
        loss_value = f'loss = label - y_prediction = {loss}'
        annotations = v + '\n' + z_formula + '\n' + z_value + '\n' + y_value + '\n' + loss_value
        ax.annotate(annotations,
                    xy=(0.5, 0.1),
                    xycoords='axes fraction',
                    fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8))

        # Remove duplicated labels
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique))

        plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    input_file_name = 'and.csv'

    df = pd.read_csv(input_file_name)

    training = df.iloc[:, 0:-1].values
    labels = df.iloc[:, -1].values

    model = Perceptron(training_sample=training, class_samples=labels, learning_rate=0.1, epochs=100)
    model.training()

