import numpy as np
import matplotlib.pyplot as plt


"""Represents simple NN, which can sort point on the chart into 2 categories."""


# Number of points in each category
number = 5

# Generating 2 sets with 5 points each. Each point has 2 attributes, representing 2 vectors on the graf
x1 = np.random.random(number)
x2 = x1 + [np.random.randint(10) / 10 for i in range(number)]
C1 = [x1, x2]

x1 = np.random.random(number)
x2 = x1 - [np.random.randint(10) / 10 for u in range(number)] - 0.1
C2 = [x1, x2]

weights = np.array([-0.3, 0.3])
for i in range(number):
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
