"""
Simple Neural Network. Receives 3 input signals and returns 1 output signal in the form of 1 or 0.
Fully Connected Neural Network with Step Activation Function.
"""


import numpy as np


def activation(x) -> 1 | 0:
    """Determines, if a neuron should or should not be activated, depending on an input signal strength

    Args:
        x: input signal strength
    Returns:
        1 or 0 – activated or not"""

    activated = 0 if x < 0.5 else 1

    return activated


def neural_network(s_1: int, s_2: int, s_3: int) -> 1 | 0:
    """Represents a simple Neural Network, made with 3 layers (input, hidden, output).

    Input layer
    ~~~~~~~~~~~~~
    Input layer consists of 3 neurons. Each neuron is connected to 2 neurons of the hidden layer, and has a weight for
    each connection. This layer has no activation threshold, it simply multiply inputs on its weigh, and 'sends' signal
    to the next neuron.

    Hidden layer
    ~~~~~~~~~~~~~
    Hidden layer consists of 2 neurons. Both connected to the single output neuron in the output layer. Each connection
    has its weight as well. Each connection here has an activation threshold, which will be calculated after weights are
    applied. If the threshold is not exceeded – neuron sends 0, otherwise it sends 1.

    Output neuron
    ~~~~~~~~~~~~~
    Output neuron receives 2 signals, each represented as integer. It sums them into 1 value and checks it for
    activation threshold, which forms the result – 1 or 0.

    Args:
        s_1: input signal to the first neuron
        s_2: input signal to the second neuron
        s_3: input signal to the third neuron
    Returns:
        1 or 0 – represents Neuron Network response"""

    input_layer = np.array([s_1, s_2, s_3])
    hidden_neuron_1_weights = [0.3, 0.3, 0]   # Weights for connection from input layer to hidden neuron 1
    hidden_neuron_2_weights = [0.4, -0.5, 1]  # Weights for connection from input layer to hidden neuron 2
    input_layer_weights = np.array([hidden_neuron_1_weights, hidden_neuron_2_weights])  # Matrix 2x3
    hidden_layer_weights = np.array([-1, 1])  # Vector 1х2

    input_layer_outputs = np.dot(input_layer_weights, input_layer)
    print(f'Input layer result signals: {input_layer_outputs}')

    hidden_layer_output = np.array([activation(x) for x in input_layer_outputs])
    print(f'Hidden layer result signals: {hidden_layer_output}')

    output_layer_input = np.dot(hidden_layer_weights, hidden_layer_output)  # int, as arrays are 1D
    output_layer_output = activation(output_layer_input)
    print(f'Output value of NN: {output_layer_output}')

    return output_layer_output


signal_1 = 1
signal_2 = 0
signal_3 = 1

response = neural_network(signal_1, signal_2, signal_3)
if response == 1:
    print("Positive response")
else:
    print("Negative response")
