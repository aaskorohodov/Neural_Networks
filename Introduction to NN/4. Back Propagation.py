import numpy as np


def ht_activation(x: np.float64) -> float:
    """Hyperbolic tangent activation function. Normalizes neurons output between 1 and 0."""

    res = 2/(1 + np.exp(-x)) - 1

    return res


def derivative_of_activation(x: np.float64 or np.array) -> float:
    """Calculates the rate of change of the output of the neuron with respect to its input

    In other words, it indicates how much the output of the neuron will change in response to a small change in its
    input.

    Args:
        x: neuron output, neurons activation. One or 2 values (one neuron ore several)
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
    for k in range(training_circles):
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
        if k % 100 == 0:
            print(error, derivative_of_activation(nn_result))

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
training_circles = 100000

# Weights
input_weights = np.array([[-0.2, 0.3, -0.4], [0.1, -0.3, -0.4]])
hidden_weights = np.array([0.2, 0.3])

train(training_set)

for case in training_set:
    input_from_training_set = case[0:3]
    result, _ = go_forward(input_from_training_set)
    print(f"Trained result: {result}; Correct result: {case[-1]}")
