
# SINGLE LAYER PERCEPTRON

import numpy as np

# Training Dataset
# [Study Hours, Attendance]
X = np.array([
    [1, 40],
    [2, 45],
    [2, 50],
    [3, 55],
    [4, 65],
    [5, 70],
    [5, 80],
    [6, 90]
], dtype=float)

# Expected Outputs
# 0 = Fail, 1 = Pass
Y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Normalize inputs
X[:, 0] = X[:, 0] / 6
X[:, 1] = X[:, 1] / 100

# Initialize weights and bias
weights = np.zeros(2)
bias = 0

learning_rate = 0.1
epochs = 100


# Step Activation Function
def activation(z):
    return 1 if z >= 0 else 0


# Training
for epoch in range(epochs):

    total_error = 0

    for i in range(len(X)):

        # Weighted sum
        z = np.dot(X[i], weights) + bias

        # Predicted output
        prediction = activation(z)

        # Calculate error
        error = Y[i] - prediction

        # Update weights
        weights += learning_rate * error * X[i]

        # Update bias
        bias += learning_rate * error

        total_error += abs(error)

    print("Epoch:", epoch + 1,
          "Total Error:", total_error)

    # Stop if no errors
    if total_error == 0:
        break


# Testing
print("\nTRAINED WEIGHTS:", weights)
print("TRAINED BIAS:", bias)

print("\nFINAL PREDICTIONS")
print("Study  Attendance  Result")

for i in range(len(X)):

    z = np.dot(X[i], weights) + bias

    prediction = activation(z)

    result = "PASS" if prediction == 1 else "FAIL"

    print(X[i, 0] * 6, "   ",
          X[i, 1] * 100, "     ", result)


# User Input
print("\nTEST NEW STUDENT")

hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance percentage: "))

# Normalize
test = np.array([hours / 6, attendance / 100])

z = np.dot(test, weights) + bias

prediction = activation(z)

if prediction == 1:
    print("Student Result: PASS")
else:
    print("Student Result: FAIL")
