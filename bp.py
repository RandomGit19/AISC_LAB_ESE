
# BACKPROPAGATION USING MULTILAYER PERCEPTRON

import numpy as np
import matplotlib.pyplot as plt


# Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of Sigmoid
def dsigmoid(y):
    return y * (1 - y)


# XOR Dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

T = np.array([0, 1, 1, 0])


# Network Parameters
input_neurons = 2
hidden_neurons = 2
output_neurons = 1

lr = 0.5
max_epochs = 10000
target_errors = 0.01


# Initialize Weights and Biases
np.random.seed(42)

w1 = np.random.uniform(
    -1, 1, (input_neurons, hidden_neurons)
)

b1 = np.zeros((1, hidden_neurons))

w2 = np.random.uniform(
    -1, 1, (hidden_neurons, output_neurons)
)

b2 = np.zeros((1, output_neurons))


error_history = []


# TRAINING
for epoch in range(1, max_epochs + 1):

    total_errors = 0

    for i in range(len(X)):

        x_i = X[i:i+1]
        t_i = T[i:i+1]

        # FORWARD PROPAGATION

        net_h = np.dot(x_i, w1) + b1
        out_h = sigmoid(net_h)

        net_o = np.dot(out_h, w2) + b2
        out_o = sigmoid(net_o)


        # ERROR CALCULATION

        error = t_i - out_o
        total_errors += np.sum(error ** 2)


        # BACKPROPAGATION

        delta_o = error * dsigmoid(out_o)

        delta_h = np.dot(delta_o, w2.T) * dsigmoid(out_h)


        # UPDATE WEIGHTS AND BIASES

        w2 += lr * np.dot(out_h.T, delta_o)
        b2 += lr * delta_o

        w1 += lr * np.dot(x_i.T, delta_h)
        b1 += lr * delta_h


    # Store Error
    error_history.append(total_errors)


    # Early Stopping
    if total_errors < target_errors:

        print("Training complete in", epoch, "epochs")
        break


# PREDICTION FUNCTION
def predict(X):

    out_h = sigmoid(np.dot(X, w1) + b1)

    out_o = sigmoid(np.dot(out_h, w2) + b2)

    return out_o


# DISPLAY RESULTS
print("\nXOR Predictions\n")

for i in range(len(X)):

    output = predict(X[i:i+1])

    print(
        X[i],
        "->",
        np.around(output[0][0], 3)
    )


# PLOT ERROR GRAPH
plt.figure(figsize=(10, 6))

plt.plot(error_history, label="Total Error")

plt.xlabel("Epochs")
plt.ylabel("Sum Squared Error")
plt.title("Error vs Epochs")

plt.grid(True)
plt.legend()
plt.show()
