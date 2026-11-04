# Training data
data = [[1, 1], [2, 1], [5, 5], [6, 5]]

# Target classes
target = [0, 0, 1, 1]

# Initial weights
weights = [[2, 2], [4, 4]]

# Class of each neuron
classes = [0, 1]

lr = 0.1

for epoch in range(100):

    for i in range(4):

        x = data[i]
        t = target[i]

        # Find nearest neuron
        min_dist = 999999
        winner = 0

        for j in range(2):

            d = (x[0] - weights[j][0]) ** 2
            d = d + (x[1] - weights[j][1]) ** 2

            if d < min_dist:
                min_dist = d
                winner = j

        # Update weights
        if classes[winner] == t:

            weights[winner][0] = weights[winner][0] + lr * (x[0] - weights[winner][0])

            weights[winner][1] = weights[winner][1] + lr * (x[1] - weights[winner][1])

        else:

            weights[winner][0] = weights[winner][0] - lr * (x[0] - weights[winner][0])

            weights[winner][1] = weights[winner][1] - lr * (x[1] - weights[winner][1])


# Display final weights
print("Final weights:")

for i in range(2):
    print("Class:", classes[i], "Weights:", weights[i])