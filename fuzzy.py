# Triangular membership function
def triangle(x, a, b, c):

    if x < a or x > c:
        return 0

    elif x == b:
        return 1

    elif x < b:

        if a == b:
            return 1

        return (x-a) / (b-a)

    else:

        if b == c:
            return 1

        return (c-x) / (c-b)


# Input temperature
temp = float(input("Enter temperature: "))


# 1. FUZZIFICATION

low = triangle(temp, 0, 0, 20)
medium = triangle(temp, 10, 20, 30)
high = triangle(temp, 20, 40, 40)


print("Low:", low)
print("Medium:", medium)
print("High:", high)


# 2. RULE EVALUATION

slow = low
normal = medium
fast = high


# 3. AGGREGATION

numerator = 0
denominator = 0

for i in range(101):

    # Output membership values
    s = triangle(i, 0, 0, 50)
    m = triangle(i, 25, 50, 75)
    f = triangle(i, 50, 100, 100)

    # Apply minimum for each rule
    r1 = min(slow, s)
    r2 = min(normal, m)
    r3 = min(fast, f)

    # Combine rules using maximum
    output = max(r1, r2, r3)

    numerator = numerator + i * output
    denominator = denominator + output


# 4. DEFUZZIFICATION

if denominator != 0:
    result = numerator / denominator
    print("Fan Speed:", result, "%")
else:
    print("No output")