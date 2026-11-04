import numpy as np

# Original bipolar pattern (5x5)
pattern = np.array([
    1,  1,  1,  1,  1,
    1, -1, -1, -1,  1,
    1, -1,  1, -1,  1,
    1, -1, -1, -1,  1,
    1,  1,  1,  1,  1
])

n = len(pattern)

# Initialize weight matrix
weights = np.zeros((n, n))

# Hebbian learning
for i in range(n):
    for j in range(n):
        if i != j:
            weights[i][j] = pattern[i] * pattern[j]

weights = weights / n

# Introduce noise
noisy_pattern = pattern.copy()

noisy_pattern[2] *= -1
noisy_pattern[7] *= -1
noisy_pattern[17] *= -1


# Recall function
def recall(input_pattern, iterations=20):
    output = input_pattern.copy()

    for _ in range(iterations):
        for i in range(n):
            activation = np.dot(weights[i], output)

            if activation >= 0:
                output[i] = 1
            else:
                output[i] = -1

    return output


# Perform recall
recalled = recall(noisy_pattern)


# Display pattern in 5x5 format
def display(title, p):
    print("\n" + title)
    print("-" * 15)

    for i in range(0, n, 5):
        row = p[i:i+5]
        print(" ".join("+" if x == 1 else "-" for x in row))


# Terminal output
print("\n========== HOPFIELD NETWORK ==========")

display("Original Pattern:", pattern)
display("Noisy Pattern:", noisy_pattern)
display("Recalled Pattern:", recalled)

print("\n========== RESULT ==========")

if np.array_equal(pattern, recalled):
    print("Pattern successfully recovered!")
else:
    print("Pattern recovery unsuccessful.")

print("======================================")