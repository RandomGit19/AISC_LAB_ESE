
import matplotlib.pyplot as plt

# Training data
data = [[1, 1], [2, 1], [5, 5], [6, 5]]

# Initial weights of 3 neurons
weights = [[0, 0], [4, 4], [8, 8]]

# Save initial weights for plotting
initial_weights = [w.copy() for w in weights]

lr = 0.1

# SOM Training
for epoch in range(100):

    for x in data:

        # Find Best Matching Unit
        min_dist = 999999
        winner = 0

        for i in range(3):

            d = (x[0] - weights[i][0]) ** 2
            d = d + (x[1] - weights[i][1]) ** 2

            if d < min_dist:
                min_dist = d
                winner = i

        # Update winner and neighbours
        for i in range(3):

            if abs(i - winner) <= 1:

                weights[i][0] += lr * (
                    x[0] - weights[i][0]
                )

                weights[i][1] += lr * (
                    x[1] - weights[i][1]
                )


# Display final weights
print("Final weights:")

for i in range(3):
    print(weights[i])


# =================================
# PLOTTING
# =================================

# Extract coordinates
x_data = [p[0] for p in data]
y_data = [p[1] for p in data]

x_initial = [p[0] for p in initial_weights]
y_initial = [p[1] for p in initial_weights]

x_final = [p[0] for p in weights]
y_final = [p[1] for p in weights]


plt.figure(figsize=(8, 6))

# Plot training data
plt.scatter(
    x_data, y_data,
    color='blue',
    label='Training Data',
    s=80
)

# Plot initial neuron weights
plt.scatter(
    x_initial, y_initial,
    color='orange',
    marker='s',
    label='Initial Weights',
    s=100
)

# Plot final neuron weights
plt.scatter(
    x_final, y_final,
    color='red',
    marker='*',
    label='Final Weights',
    s=200
)

# Connect final neurons
plt.plot(
    x_final, y_final,
    'r--',
    label='SOM Neuron Connections'
)

plt.title("Self Organizing Map (SOM)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.legend()
plt.grid(True)
plt.show()
