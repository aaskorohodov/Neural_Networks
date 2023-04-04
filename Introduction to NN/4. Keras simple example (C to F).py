"""Keras is a wrapper, made for TenserFlow.


TenserFlow requires CUDA, to run on GPU, but can work without it as well, simply running on CPU. But if there is no
CUDA installed, TenserFlow will print warnings. To avoid these warnings:


    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


This example trains NN to convert temperature from C to F. We will use 1 input neuron with and 1 Output, overall
2 neurons with 1 connection. Also, there is going to be a bias, to represent actual equation, that is used to convert
C to F:

    F = C * 1.8 + 32

Bias represents +32, and is has its own weight. So overall, NN will have 2 weights:

    1. Input-Output
    2. Bias-Output

*Bias is treated as a fully engaged signal (for example +1), onto which weight is applied.
So overall, NN will have to learn 2 weights."""


import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Training set
c = np.array([-40, -10, 0, 8, 15, 22, 38])
f = np.array([-40, 14, 32, 46, 59, 72, 100])

# Setting model as sequential, i.e. consisting of layers going one after another.
model = keras.Sequential()

# Adding 1 input neuron (units) with 1 input (input_shape) and setting Activation Function to linear
model.add(Dense(units=1, input_shape=(1,), activation='linear'))

# Setting quality criterion and gradient decent optimization algorithm
learning_rate = 0.1
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(learning_rate))

'''At this point, initial weights are being set automatically, bias is being created automatically as well'''

# Starting training NN. verbose=False will prevent printing current state into console
log = model.fit(c, f, epochs=5000, verbose=False)

plt.plot(log.history['loss'])
plt.grid(True)
plt.show()

# Checking, what model can do after training. *100C == ~212F
print(model.predict([100]))
print(model.get_weights())
