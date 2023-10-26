# Perceptron em Python

This repository contains an implementation of the Perceptron algorithm in Python for learning purposes.

## Description

Perceptron is one of the oldest and most fundamental algorithms in machine learning. The perceptron in its conception would be a supervised logistic regression algorithm. The perceptron uses an activation function that produces a binary output (1 or 0) based on a weighted sum of the inputs. If the weighted sum is greater than a certain threshold it classifies it as 1, otherwise it classifies it as 0. Therefore we can see the perceptron as a simplified form of binary logistic regression, where the activation function is a step function.

## Characteristics

- Implementation of Perceptron (with aditional of bias and randomic weights)
- Visualization of the decision boundary at each iteration.
- Detailed graph annotations to understand the training process.

## Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib

## How to use

1. Clone this repository.
2. Make sure you have all necessary libraries installed.
3. Add your dataset in CSV format. The example provided uses a file called `and.csv`. 
4. The dataset must 2 features and 1 class, because the plot will work only for this kind of data (Remember this code is for study purpose)
4. Run the script to train the Perceptron and visualize the decision boundary.

## License

This project is under the MIT License. See the `LICENSE` file in the root directory for details.

## Contact

Thiago Seronni Mendonça - [thiagoseronni@gmail.com](mailto:thiagoseronni@gmail.com)