import numpy as np

# Training patterns (bipolar values)
X = np.array([
    [1, -1, 1],
    [-1, 1, -1]
])

Y = np.array([
    [1, 1],
    [-1, -1]
])

# Initialize weight matrix
weights = np.zeros((X.shape[1], Y.shape[1]))

# Hebbian learning
for i in range(len(X)):
    weights += np.outer(X[i], Y[i])

# Normalize weights
weights = weights / len(X)


# BAM recall function
# BAM recall function
def recall(input_pattern, direction="XY", iterations=10):

    output = input_pattern.copy()

    for _ in range(iterations):

        if direction == "XY":

            # Forward: X -> Y
            activation = np.dot(output, weights)
            y = np.where(activation >= 0, 1, -1)

            # Backward: Y -> X
            activation = np.dot(y, weights.T)
            new_output = np.where(activation >= 0, 1, -1)

        else:

            # Backward: Y -> X
            activation = np.dot(output, weights.T)
            x = np.where(activation >= 0, 1, -1)

            # Forward: X -> Y
            activation = np.dot(x, weights)
            new_output = np.where(activation >= 0, 1, -1)

        # Check whether the network has stabilized
        if np.array_equal(output, new_output):
            break

        output = new_output

    # Return the recalled pattern in the requested direction
    if direction == "XY":
        return np.where(np.dot(output, weights) >= 0, 1, -1)
    else:
        return np.where(np.dot(output, weights.T) >= 0, 1, -1)


# Display function
def display(title, pattern):
    print("\n" + title)
    print("-" * 30)

    for row in pattern:
        print(" ".join("+" if x == 1 else "-" for x in row))


print("\n========== BAM NETWORK ==========")

# Display training patterns
display("Input Patterns (X):", X)
display("Associated Patterns (Y):", Y)

# Recall Y from X
test_X = X[0].copy()
recalled_Y = recall(test_X, "XY")

print("\nForward Recall:")
print("Input X:       ", test_X)
print("Recalled Y:    ", recalled_Y)
print("Expected Y:    ", Y[0])

# Recall X from Y
test_Y = Y[1].copy()
recalled_X = recall(test_Y, "YX")

print("\nBackward Recall:")
print("Input Y:       ", test_Y)
print("Recalled X:    ", recalled_X)
print("Expected X:    ", X[1])

print("\n=================================")