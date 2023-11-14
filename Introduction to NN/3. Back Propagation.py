"""
This is a basic example of BackPropagation – a process of teaching NN, by calculation overall error of NN's output,
and modifying weights of those neurons, that are responsible for that error more.

        Error
~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, there is 1 neuron in the output layer, its output is normalized between -1 and 1 (returns a float).
The goal of this NN is to indicate, if some input-data is good or bad, in other words – to sort input-data in 1 of 2
categories.

The error is simply the difference between expected and actual results. For example:

    input-data -> result == -1
    expected_result == 1
    error = 1 - (-1) == 2 (big error)

    input-data -> result == -1
    expected_result == -1
    error = -1 - (-1) == 0 (no error)

    input-data -> result == 0.7
    expected_result == 1
    error = 0.7 - 1 == 0.3 (small error)

        Activation function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hyperbolic tangent activation function is used as an activation function. It normalizes each neuron's output from
-1 to 1. Normalization is required, NOT to get very strong signals from some neurons, which in practice will result in
unpredictable response from NN and will make learning harder and longer.

        Derivative of activation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Derivative of the activation function indicates, how much the change of the input will affect function's result. For
example, if a function's derivative is 100500, than a small change (for example, change in the input by 1) will result
in the output change by 100500. This means that this specific argument (there may be several arguments), in the value
that this argument is right now, will result in a huge change of the output, if this argument will be changed.

Derivative of activation is used in NN, to get understanding about how much should we modify weight of a specific
connection. The bigger derivative is – the stronger its effect on the overall result will be, and the more we will
change that weight.

        BackPropagation, the essence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BackPropagation begins from the last neuron, for which we calculate the error. After we do so, we can then get the
idea, of which inputs played the most in that error. For example, one of 2 inputs may have a big weight and its
input is big (0.9), while the second one has low weight and a very small input (0.001). This will lead us to the first
neuron, which played the most role in the overall error. This neuron's weight will be modified significantly more,
then weight of another neuron, which makes sense – 'you generated the most part of the error, so your weight will be
modified more'.

Then, all we have to do, is keep traveling backwards from the last to the first, layer by layer, keeping in mind the
overall error, that NN made. Having this error, we will keep calculating derivative of activation to each neuron, and
the contribution of each to the overall NN result. This means that for each neuron we will not only get the idea,
of now much this exact neuron contributed to the error, but also we will understand, how much this neuron will react
to the change in the input. If some neuron will react a lot, and it made big contribution – its weights will be modified
more.

This is the essence of BackPropagation – we are getting from the last neuron to the first one, and modifying weight
of those connections, which contributed more to the overall error, and making this modification more, the higher
overall error is. This 2 rules help us with:

    1. NOT modify weights/make very small changes, if the error is small or if there is no error
    2. Modify weights of those connections, which are responsible more for the error
"""


import numpy as np
from typing import NewType


FloatZeroToOne = NewType('FloatZeroToOne', float)


def ht_activation(x: np.float64) -> FloatZeroToOne:
    """Hyperbolic tangent activation function. Normalizes neurons output between 0 and 1."""

    res = 2/(1 + np.exp(-x)) - 1

    return res


def derivative_of_activation(x: np.float64 or np.array) -> float:
    """Calculates the rate of change of the output of the neuron with respect to its input

    In other words, it indicates how much the output of the neuron will change in response to a small change in its
    input.

    Args:
        x: neuron output, neurons activation. One or 2 values (one neuron or several)
    Returns:
        derivative of activation function"""

    res = 0.5*(1 + x)*(1 - x)

    return res


def go_forward(input_data: tuple[int, int, int]) -> tuple[float, np.array]:
    """Launches NN in regular way. Except – returns not only result but signals to hidden layer as well.

    Args:
        input_data: tuple with input data for the first neuron layer
    Returns:
        tuple(NN_result, input->hidden_layer_signals)"""

    hidden_layer_pre_activation = np.dot(input_weights, input_data)
    hidden_layer_activation = np.array([ht_activation(x) for x in hidden_layer_pre_activation])

    # noinspection PyTypeChecker
    result_pre_activation: np.float64 = np.dot(hidden_weights, hidden_layer_activation)
    result_activation = ht_activation(result_pre_activation)

    return result_activation, hidden_layer_activation


def train(input_data):
    global hidden_weights, input_weights

    input_data_len = len(input_data)
    for k in range(training_cycles):
        # Selecting random input data from training set
        x = input_data[np.random.randint(0, input_data_len)]
        correct_result = x[-1]
        input_set = x[0:3]

        # Launching NN as usual
        nn_result, hidden_layer_results = go_forward(x[0:3])

        # Overall error of NN, aka the value, representing how much NN missed
        error = nn_result - correct_result

        '''Delta is a local gradient'''
        # Local gradient and values correction for hidden neurons weights ('hidden -> output layer' connection)
        delta = error * derivative_of_activation(nn_result)  # Local gradient
        neuron_0_weight_correction = learning_rate * delta * hidden_layer_results[0]
        neuron_1_weight_correction = learning_rate * delta * hidden_layer_results[1]

        # Correcting weights of the 'hidden_neurons -> output_layer' connection
        hidden_weights[0] -= neuron_0_weight_correction    # Hidden weight #1 correction
        hidden_weights[1] -= neuron_1_weight_correction    # Hidden weight #2 correction
        # if k % 100 == 0:
        #     print(error, derivative_of_activation(nn_result))

        # Correcting weights of the 'input_layer -> hidden_layer' connection
        # Vector of 2 values of local gradients
        delta2 = hidden_weights * delta * derivative_of_activation(hidden_layer_results)
        input_layer_correction_0 = np.array(input_set) * delta2[0] * learning_rate
        input_layer_correction_1 = np.array(input_set) * delta2[1] * learning_rate

        input_weights[0, :] = input_weights[0, :] - input_layer_correction_0
        input_weights[1, :] = input_weights[1, :] - input_layer_correction_1


# Training sample (aka full sample)
training_set = [(-1, -1, -1, -1),
                (-1, -1, 1, 1),
                (-1, 1, -1, -1),
                (-1, 1, 1, 1),
                (1, -1, -1, -1),
                (1, -1, 1, 1),
                (1, 1, -1, -1),
                (1, 1, 1, -1)]
# Learning rate
learning_rate = 0.01
# Learning cycles. Aka, how many learning iterations there would be
training_cycles = 100000

# Weights, randomly selected as some initial weights
input_weights = np.array([[-0.2, 0.3, -0.4], [0.1, -0.3, -0.4]])
hidden_weights = np.array([0.2, 0.3])

train(training_set)

for case in training_set:
    input_from_training_set = case[0:3]
    result, _ = go_forward(input_from_training_set)
    print(f"Trained result: {result}; Correct result: {case[-1]}")
