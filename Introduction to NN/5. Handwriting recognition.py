import keras

from keras.datasets import mnist
from keras.layers import Dense, Flatten
from matplotlib import pyplot as plt

"""

Each image has is 28x28 and made in b/w, so each pixel has a value from 0 (black) to 255 (white)
"""

"""
x_train – images of numbers, training set
y_train – vector of recognised images from training set (correct answers)
x_test – images of numbers, test set
y_test – vector of recognised images from test set (correct answers)
"""
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Standardization from 0 to 1 (because each pixel is from 0 to 255)
x_train = x_train / 255
x_test = x_test / 255

"""Right now, y_train is an array, made from correct answers, where each value is a digit from 0 to 9. It simply
indicates, what number is written in a training set. We will build a NN with 10 output neurons, so the output signal
will be an array of 10 values:
    
    result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] – answer is 9
    result = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] – answer is 0
    result = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] – answer is 5
    result = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] – answer is 1

Here, each index will indicate the output signal for each output neuron. So, 10 neurons == 10 values in a vector. What
we need is to turn each correct answer from y_train and y_test into such vector:

    answer is 9 => [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    answer is 0 => [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ...

For that purpose, there is a method right inside keras. Here, the last param (10) is a size of each vector"""
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# Printin first 25 images from training set
plt.figure(figsize=(10, 5))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    # noinspection PyUnresolvedReferences
    plt.imshow(x_train[i], cmap=plt.cm.binary)

plt.show()

model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

"""Setting other settings of the model:

    Adam – optimization algorithm, that determines the direction of weight modification and automatically calculates
    Lambda (the amount of modification). It does all of that for each weight and bias in each neuron.
    
    Categorical cross-entropy – the loss function, that calculates the amount of error. In this example, it takes
    results of all output layer, which is a vector of 10 values, and calculates it into a single value, representing
    an error.
    
    Metrics – list/dict of metrics, that user wants to see. In this example this will be an overall error."""
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Printin NN's structure
print(model.summary())
