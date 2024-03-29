"""
Represents simple NN, that can sort point on the chart into 2 categories.

If visualized in a simple 2D-graph, input data could be represented as points on that graph. This NN can sort these
points, using some separation line, that can also be drawn on a graph - all points on one side of this line will be
sorted into one category, while points from another side will be sorted into another category.
"""


import numpy as np
import matplotlib.pyplot as plt


# Number of points in each category
number_of_points = 5

# Generating 2 sets with 5 points each. Each point has 2 attributes, representing 2 coordinates on the graph
x1 = np.random.random(number_of_points)
x2 = x1 + [np.random.randint(10) / 10 for i in range(number_of_points)]
C1 = [x1, x2]

x1 = np.random.random(number_of_points)
x2 = x1 - [np.random.randint(10) / 10 for u in range(number_of_points)] - 0.1
C2 = [x1, x2]

# In this example we are feeding only points from C2-class, and checking, if NN does recognize them as C2
weights = np.array([-0.3, 0.3])
for i in range(number_of_points):
    x = np.array([C2[0][i], C2[1][i]])
    y = np.dot(weights, x)
    if y >= 0:
        print("Класс C1")
    else:
        print("Класс C2")

plt.scatter(C1[0][:], C1[1][:], s=10, c='red')
plt.scatter(C2[0][:], C2[1][:], s=10, c='blue')
separation_line = [0, 1]
plt.plot(separation_line)
plt.grid(True)
plt.show()
