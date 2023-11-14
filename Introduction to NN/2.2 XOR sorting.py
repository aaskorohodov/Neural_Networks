"""
XOR sorting, represented on 2D graph, are 2 separation line. All points between these lines is one class, and it
should be sorted into first category, while all points aside these lines should be sorted into second category.

To do these, we will need 3 layers, where input and hidden layers will have 2 neurons each, and 1 neuron in the
output layer. We can also use a bias, to move these lines.
"""


import numpy as np


def activation(input_signal) -> int:
    """Activation-function

    Args:
        input_signal: input signal
    Returns:
        Output signal (1 or 0)"""

    return 0 if input_signal <= 0 else 1


def go(points: tuple, bias: int) -> int:
    """Classification of points

    Args:
        points: Tuple with 2 coordinates
        bias: Bias, to move our line
    Returns:
        1 for sorting into 1 category, otherwise 0"""

    x = np.array([points[0], points[1], bias])
    w1 = [1, 1, -1.5]
    w2 = [1, 1, -0.5]
    w_hidden = np.array([w1, w2])
    w_out = np.array([-1, 1, -0.5])

    sum_hidden = np.dot(w_hidden, x)
    out = [activation(x) for x in sum_hidden]
    out.append(1)
    out = np.array(out)

    sum_out = np.dot(w_out, out)
    y = activation(sum_out)
    return y


c_1 = [(1, 0), (0, 1)]
c_2 = [(0, 0), (1, 1)]
b = 1
print(go(c_1[0], b), go(c_1[1], b))
print(go(c_2[0], b), go(c_2[1], b))
