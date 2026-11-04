# 1. WATER JUG PROBLEM USING BFS

from collections import deque

def water_jug(cap1, cap2, target):
    start = (0, 0)
    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        (x, y), path = queue.popleft()

        if x == target or y == target:
            print("\nSolution Found!")
            for state in path:
                print(state)
            print("Total Steps:", len(path) - 1)
            return

        # Generate all possible operations
        next_states = [
            (cap1, y),                  # Fill Jug 1
            (x, cap2),                  # Fill Jug 2
            (0, y),                     # Empty Jug 1
            (x, 0),                     # Empty Jug 2
            (x - min(x, cap2-y),        # Pour Jug 1 into Jug 2
             y + min(x, cap2-y)),
            (x + min(y, cap1-x),        # Pour Jug 2 into Jug 1
             y - min(y, cap1-x))
        ]

        for state in next_states:
            if state not in visited:
                visited.add(state)
                queue.append((state, path + [state]))

    print("No solution possible!")

# Main
cap1 = 4
cap2 = 3
target = 2

water_jug(cap1, cap2, target)