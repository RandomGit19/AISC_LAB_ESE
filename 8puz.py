# 2. A* ALGORITHM FOR 8 PUZZLE

import heapq

start = (1, 2, 3,
         4, 0, 6,
         7, 5, 8)

goal = (1, 2, 3,
        4, 5, 6,
        7, 8, 0)


# Manhattan Distance Heuristic
def heuristic(state):
    distance = 0

    for i in range(9):
        value = state[i]

        if value != 0:
            current_row = i // 3
            current_col = i % 3

            goal_index = goal.index(value)

            goal_row = goal_index // 3
            goal_col = goal_index % 3

            distance += abs(current_row - goal_row)
            distance += abs(current_col - goal_col)

    return distance


# Generate neighbouring states
def get_neighbors(state):
    neighbors = []

    zero = state.index(0)
    row, col = zero // 3, zero % 3

    moves = [
        (-1, 0), (1, 0),
        (0, -1), (0, 1)
    ]

    for dr, dc in moves:
        nr, nc = row + dr, col + dc

        if 0 <= nr < 3 and 0 <= nc < 3:
            new_pos = nr * 3 + nc

            new_state = list(state)

            new_state[zero], new_state[new_pos] = (
                new_state[new_pos], new_state[zero]
            )

            neighbors.append(tuple(new_state))

    return neighbors


# A* Algorithm
def astar(start):
    pq = []

    heapq.heappush(pq, (heuristic(start), 0, start))

    parent = {start: None}
    cost = {start: 0}

    while pq:
        f, g, current = heapq.heappop(pq)

        if g != cost[current]:
            continue

        if current == goal:
            path = []

            while current is not None:
                path.append(current)
                current = parent[current]

            path.reverse()
            return path

        for neighbor in get_neighbors(current):

            new_cost = g + 1

            if neighbor not in cost or new_cost < cost[neighbor]:

                cost[neighbor] = new_cost
                parent[neighbor] = current

                new_f = new_cost + heuristic(neighbor)

                heapq.heappush(
                    pq, (new_f, new_cost, neighbor)
                )

    return None


# Display Puzzle
def display(state):
    for i in range(0, 9, 3):
        print(state[i:i+3])
    print()


# Main
print("Initial State:")
display(start)

path = astar(start)

if path:
    print("Solution Found!")

    for step, state in enumerate(path):
        print("Step:", step)
        display(state)

    print("Total Moves:", len(path) - 1)

else:
    print("No Solution Found!")