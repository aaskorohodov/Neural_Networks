"""
Represents simple NN, that can sort point on the chart into 2 categories, with bias.

In case of perceptron, bias will simply move the separation line on 2D-graph up or down.
"""


import numpy as np
import matplotlib.pyplot as plt


# Number of points in each category
number_of_points = 5
bias = 3

# Generating 2 sets with 5 points each. Each point has 2 attributes, representing 2 coordinates on the graph
x1 = np.random.random(number_of_points)
x2 = x1 + [np.random.randint(10) / 10 for i in range(number_of_points)] + bias
C1 = [x1, x2]

x1 = np.random.random(number_of_points)
x2 = x1 - [np.random.randint(10) / 10 for u in range(number_of_points)] - 0.1 + bias
C2 = [x1, x2]

# In this example we are feeding only points from C2-class, and checking, if NN does recognize them as C2
w2 = 0.5
w3 = -bias * w2
weights = np.array([-w2, w2, w3])
for i in range(number_of_points):
    x = np.array([C2[0][i], C2[1][i], 1])
    y = np.dot(weights, x)
    if y >= 0:
        print("Класс C1")
    else:
        print("Класс C2")

plt.scatter(C1[0][:], C1[1][:], s=10, c='red')
plt.scatter(C2[0][:], C2[1][:], s=10, c='blue')
separation_line = [0 + bias, 1 + bias]
plt.plot(separation_line)
plt.grid(True)
plt.show()
