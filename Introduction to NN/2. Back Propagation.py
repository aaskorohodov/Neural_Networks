import numpy as np


def ht_activation(x: np.array) -> float:
    """Hyperbolic tangent activation function. Normalizes neurons output between 1 and 0."""

    result = 2/(1 + np.exp(-x)) - 1

    return result


def derivative_of_activation(x: int or float) -> float:
    """The purpose of this function is to provide a smooth gradient"""

    result = 0.5*(1 + x)*(1 - x)

    return result


def go_forward(input_data: tuple[int, int, int]) -> tuple[float, np.array]:
    """Launches NN in regular way. Except – returns not only result but signals to hidden layer as well.

    Args:
        input_data: tuple with input data for the first neuron layer
    Returns:
        tuple(NN_result, input->hidden_layer_signals)"""

    hidden_layer_pre_activation = np.dot(input_weights, input_data)
    hidden_layer_activation = np.array([ht_activation(x) for x in hidden_layer_pre_activation])

    result_pre_activation = np.dot(hidden_weights, hidden_layer_activation)
    result_activation = ht_activation(result_pre_activation)

    return result_activation, hidden_layer_activation


def train(input_data):
    global hidden_weights, input_weights

    input_data_len = len(input_data)
    for k in range(training_circles):
        # Selecting random input data from training set
        x = input_data[np.random.randint(0, input_data_len)]
        correct_result = x[-1]
        input_set = x[0:3]

        # Launching NN as usual
        result, hidden_layer_results = go_forward(x[0:3])

        error = result - correct_result

        # Local gradient and values correction for hidden neurons weights ('hidden -> output layer' connection)
        delta = error * derivative_of_activation(result)
        neuron_0_weight_correction = lambda_ * delta * hidden_layer_results[0]
        neuron_1_weight_correction = lambda_ * delta * hidden_layer_results[1]

        # Correcting weights of the 'hidden_neurons -> output_layer' connection
        hidden_weights[0] -= neuron_0_weight_correction    # Hidden weight #1 correction
        hidden_weights[1] -= neuron_1_weight_correction    # Hidden weight #2 correction

        # Correcting weights of the 'input_layer -> hidden_layer' connection
        # Vector of 2 values of local gradients
        delta2 = hidden_weights * delta * derivative_of_activation(hidden_layer_results)
        input_layer_correction_0 = np.array(input_set) * delta2[0] * lambda_
        input_layer_correction_1 = np.array(input_set) * delta2[1] * lambda_

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
lambda_ = 0.01
# Learning cycles. Aka, how many learning iterations there would be
training_circles = 100000

# Weights
input_weights = np.array([[-0.2, 0.3, -0.4], [0.1, -0.3, -0.4]])
hidden_weights = np.array([0.2, 0.3])

train(training_set)

for case in training_set:
    input_from_training_set = case[0:3]
    nn_result, _ = go_forward(input_from_training_set)
    print(f"Trained result: {nn_result}; Correct result: {case[-1]}")
