import numpy as np


def ht_activation(x):
    """Hyperbolic tangent activation function. Normalizes neurons output between 1 and 0."""

    result = 2/(1 + np.exp(-x)) - 1

    return result


def df(x):
    return 0.5*(1 + x)*(1 - x)


input_weights = np.array([[-0.2, 0.3, -0.4], [0.1, -0.3, -0.4]])
hidden_weights = np.array([0.2, 0.3])


def go_forward(input_data: tuple[int, int, int]):
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

    count = len(input_data)
    for k in range(training_circles):
        x = input_data[np.random.randint(0, count)]  # случайный выбор входного сигнала из обучающей выборки
        y, out = go_forward(x[0:3])             # прямой проход по НС и вычисление выходных значений нейронов
        e = y - x[-1]                           # Error
        delta = e*df(y)                         # Local gradient
        neuron_0_weight_correction = lambda_ * delta * out[0]
        neuron_1_weight_correction = lambda_ * delta * out[1]

        # Correcting weights of the hidden neurons,
        hidden_weights[0] -= neuron_0_weight_correction    # Hidden weight #1 correction
        hidden_weights[1] -= neuron_1_weight_correction    # Hidden weight #2 correction

        delta2 = hidden_weights * delta * df(out)               # вектор из 2-х величин локальных градиентов

        # корректировка связей первого слоя
        input_weights[0, :] = input_weights[0, :] - np.array(x[0:3]) * delta2[0] * lambda_
        input_weights[1, :] = input_weights[1, :] - np.array(x[0:3]) * delta2[1] * lambda_


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

train(training_set)

# Che
for case in training_set:
    input_from_training_set = case[0:3]
    y, out = go_forward(input_from_training_set)
    print(f"Выходное значение НС: {y} => {case[-1]}")
