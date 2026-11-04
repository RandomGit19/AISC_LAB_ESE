"""
AI & SOFT COMPUTING — ALL UPLOADED EXPERIMENTS (ONE PYTHON FILE)
=================================================================
Experiments: 01, 01.2, 02, 03, 05, 06, 07, 08, 09, 10.
Experiment 04: NO PDF uploaded; not invented here.

Run all:             python AISC_All_Experiments.py
Run one:             python AISC_All_Experiments.py 3
Run individual part: python AISC_All_Experiments.py 7-bam
List sections:       python AISC_All_Experiments.py --list

Dependencies: NumPy, SciPy, scikit-learn.
    pip install numpy scipy scikit-learn
Optional: scikit-fuzzy, for the original Experiment 09 library API.
    pip install scikit-fuzzy
This file avoids graph plotting because the requested Lab ESE format has no graphs.

SOURCE / SCOPE NOTES
--------------------
01, 01.2, 02: PDFs are reconstructed explainers (not official handouts).
03: source activation functions and AND/OR/NOT/XOR problem statements.
05: reference is a SINGLE numerical backpropagation example; the case studies
    specify tasks but do not provide full code. Numeric example implemented.
06: source LVQ digits example + SOM heart-failure case-study outline. The SOM
    demo uses synthetic clinical-like data, NOT actual patients or the named CSV.
07: source gives Hopfield digit and BAM bilingual problem statements, NOT code.
    Demonstrations implement the stated ideas using small offline patterns.
08: source fuzzy membership/operations example and AQI relations case study.
09: source fan-speed fuzzy controller example and traffic controller case study.
    A NumPy-only Mamdani controller is provided so it works even without skfuzzy.
10: source gives ANFIS design outline but no complete code; a compact real
    first-order Sugeno ANFIS demo uses Gaussian premise MFs, alternating least
    squares for consequent parameters and optimized premise parameters.
All non-source example data are clearly labelled illustrative/dummy data.
"""

import sys
from collections import deque
import numpy as np


def banner(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


# ============================================================================
# EXPERIMENT 01 — MISSIONARIES AND CANNIBALS (BFS)
# ============================================================================
def exp01_missionaries():
    banner("EXP 01: MISSIONARIES AND CANNIBALS — BFS")
    start, goal = (3, 3, 1), (0, 0, 0)
    loads = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)]

    def valid(s):
        m, c, _ = s
        if not (0 <= m <= 3 and 0 <= c <= 3):
            return False
        rm, rc = 3 - m, 3 - c
        return (m == 0 or m >= c) and (rm == 0 or rm >= rc)

    def neighbors(s):
        m, c, boat = s
        direction = -1 if boat == 1 else 1
        for dm, dc in loads:
            nxt = (m + direction * dm, c + direction * dc, 1 - boat)
            if valid(nxt):
                yield nxt, (dm, dc)

    queue = deque([start])
    parent = {start: None}
    move_used = {}
    while queue:
        current = queue.popleft()
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            path.reverse()
            print("Minimum boat crossings:", len(path) - 1)
            for i, s in enumerate(path):
                if i == 0:
                    print(i, "Start", s)
                else:
                    dm, dc = move_used[s]
                    direction = "RIGHT" if s[2] == 0 else "LEFT"
                    print(i, f"{dm}M {dc}C -> {direction}", s)
            return path
        for nxt, load in neighbors(current):
            if nxt not in parent:  # parent keys also act as visited set
                parent[nxt] = current
                move_used[nxt] = load
                queue.append(nxt)
    print("No solution")
    return None


# ============================================================================
# EXPERIMENT 01.2 — WATER JUG (BFS AND DFS)
# ============================================================================
def exp012_water_jug(M=3, N=4, target=2):
    banner("EXP 01.2: WATER JUG — BFS AND DFS")
    start = (0, 0)

    def successors(s):
        x, y = s
        a = min(x, N - y)
        b = min(y, M - x)
        return [
            ("Fill jug 1", (M, y)),
            ("Fill jug 2", (x, N)),
            ("Empty jug 1", (0, y)),
            ("Empty jug 2", (x, 0)),
            ("Pour jug 1 -> 2", (x - a, y + a)),
            ("Pour jug 2 -> 1", (x + b, y - b)),
        ]

    def is_goal(s):
        return target in s

    def reconstruct(parent, end):
        path = []
        while end is not None:
            prev, action = parent[end]
            path.append((action, end))
            end = prev
        return list(reversed(path))

    def bfs():
        q = deque([start])
        parent = {start: (None, "Start")}
        while q:
            s = q.popleft()
            if is_goal(s):
                return reconstruct(parent, s)
            for action, nxt in successors(s):
                if nxt not in parent:
                    parent[nxt] = (s, action)
                    q.append(nxt)
        return None

    def dfs():
        visited = set()
        parent = {start: (None, "Start")}

        def search(s):
            visited.add(s)
            if is_goal(s):
                return s
            for action, nxt in successors(s):
                if nxt not in visited:
                    parent[nxt] = (s, action)
                    result = search(nxt)
                    if result is not None:
                        return result
            return None

        end = search(start)
        return reconstruct(parent, end) if end is not None else None

    results = {}
    for name, algorithm in [("BFS", bfs), ("DFS", dfs)]:
        path = algorithm()
        results[name] = path
        print("\n", name, "solution")
        if path is None:
            print("No solution")
        else:
            for i, (action, state) in enumerate(path):
                print(i, action, "->", state)
            print("Total moves:", len(path) - 1)
    return results


# ============================================================================
# EXPERIMENT 02 — WUMPUS WORLD KNOWLEDGE-BASED AGENT
# Exact 4x4 hidden world from uploaded explainer; rules R1-R5.
# ============================================================================
def exp02_wumpus():
    banner("EXP 02: WUMPUS WORLD — SAFE KNOWLEDGE-BASED AGENT")
    size = 4
    start = (1, 1)
    wumpus = (1, 3)
    gold = (2, 3)
    pits = {(3, 1), (3, 3), (4, 4)}
    cells = {(x, y) for x in range(1, size + 1) for y in range(1, size + 1)}

    def adjacent(cell):
        x, y = cell
        # Right first, then up, to match the explanatory trace.
        return [(a, b) for a, b in [(x + 1, y), (x, y + 1),
                                     (x - 1, y), (x, y - 1)] if (a, b) in cells]

    def percept(cell):
        around = adjacent(cell)
        return {
            "breeze": any(n in pits for n in around),
            "stench": wumpus in around,
            "glitter": cell == gold,
        }

    # Agent knowledge. The agent does not read the hidden pits/wumpus directly.
    visited = set()
    observations = {}
    no_pit = set()
    no_wumpus = set()
    known_pits = set()
    known_wumpus = set()
    current = start
    travel = [start]

    def infer():
        changed = True
        while changed:
            before = (len(no_pit), len(no_wumpus), len(known_pits), len(known_wumpus))
            for cell, sensed in observations.items():
                # R1: survived visit => safe.
                no_pit.add(cell)
                no_wumpus.add(cell)
                neighbors = adjacent(cell)
                # R2: absence of breeze or stench clears all neighboring cells.
                if not sensed["breeze"]:
                    no_pit.update(neighbors)
                if not sensed["stench"]:
                    no_wumpus.update(neighbors)
                # R3 and R4: only one candidate left => identify danger.
                if sensed["breeze"]:
                    candidates = [n for n in neighbors if n not in no_pit]
                    if len(candidates) == 1:
                        known_pits.add(candidates[0])
                if sensed["stench"]:
                    candidates = [n for n in neighbors if n not in no_wumpus]
                    if len(candidates) == 1:
                        known_wumpus.add(candidates[0])
            # R5: exactly one Wumpus in this world.
            if len(known_wumpus) == 1:
                no_wumpus.update(cells - known_wumpus)
            after = (len(no_pit), len(no_wumpus), len(known_pits), len(known_wumpus))
            changed = after != before

    def safe_route(origin, target, safe):
        q = deque([origin])
        parent = {origin: None}
        while q:
            cell = q.popleft()
            if cell == target:
                path = []
                while cell is not None:
                    path.append(cell)
                    cell = parent[cell]
                return path[::-1]
            for nxt in adjacent(cell):
                if nxt in safe and nxt not in parent:
                    parent[nxt] = cell
                    q.append(nxt)
        return None

    for _ in range(100):
        if current not in visited:
            seen = percept(current)
            observations[current] = seen
            visited.add(current)
            print("At", current, "percepts:", seen)
            infer()
        if observations[current]["glitter"]:
            print("Gold grabbed at", current)
            route = safe_route(current, start, visited)
            if route is None:
                print("Cannot safely return")
                return False
            travel.extend(route[1:])
            print("Safe return path:", route)
            print("Full journey:", travel)
            print("Known pits:", sorted(known_pits), "Wumpus:", sorted(known_wumpus))
            return True
        safe = no_pit & no_wumpus
        options = []
        for destination in sorted(safe - visited):
            route = safe_route(current, destination, safe)
            if route:
                options.append(route)
        if not options:
            print("No unvisited PROVED-SAFE cell: stop (do not guess)")
            return False
        route = min(options, key=lambda p: (len(p), p[-1]))
        for step in route[1:]:
            print("Move:", current, "->", step)
            current = step
            travel.append(current)
    print("Exploration limit reached")
    return False


# ============================================================================
# EXPERIMENT 03 — ACTIVATION FUNCTIONS, PERCEPTRON, XOR
# ============================================================================
def exp03_activation():
    banner("EXP 03-A: SIX ACTIVATION FUNCTIONS (NO GRAPHS)")
    x = np.array([-2, -1, 0, 1, 2], dtype=float)

    def symmetric_hard_limit(a, threshold=2):
        return np.where(a >= threshold, 1, -1)

    def binary_step(a):
        return np.where(a >= 0, 1, 0)

    def bipolar_step(a):
        return np.where(a >= 0, 1, -1)

    def saturating_linear(a):
        return np.clip(a, -1, 1)

    def tansig(a):
        return np.tanh(a)

    def log_sigmoid(a):
        return 1 / (1 + np.exp(-a))

    print("Input:      ", x)
    for name, func in [
        ("Symmetric hard limit", symmetric_hard_limit),
        ("Binary step", binary_step),
        ("Bipolar step", bipolar_step),
        ("Saturating linear", saturating_linear),
        ("Tansig", tansig),
        ("Log sigmoid", log_sigmoid),
    ]:
        print(f"{name:22s}: {np.round(func(x), 4)}")


def exp03_perceptron():
    banner("EXP 03-B: TRAIN A SINGLE-LAYER PERCEPTRON FOR AND, OR, NOT")

    def train(X, y, lr=0.1, epochs=100):
        weights = np.zeros(X.shape[1])
        bias = 0.0
        for epoch in range(epochs):
            mistakes = 0
            for xi, target in zip(X, y):
                z = np.dot(xi, weights) + bias
                prediction = 1 if z >= 0 else 0
                error = target - prediction
                weights += lr * error * xi
                bias += lr * error
                mistakes += (error != 0)
            if mistakes == 0:
                break
        return weights, bias

    def predict(X, w, b):
        return np.where(X @ w + b >= 0, 1, 0)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    for name, y in [
        ("AND", np.array([0, 0, 0, 1])),
        ("OR", np.array([0, 1, 1, 1])),
        ("NAND", np.array([1, 1, 1, 0])),  # additional common ESE variant
        ("NOR", np.array([1, 0, 0, 0])),   # additional common ESE variant
    ]:
        w, b = train(X, y)
        output = predict(X, w, b)
        print(f"{name:5s} actual={y.tolist()} predicted={output.tolist()} w={w} b={b:.2f}")
        assert np.array_equal(y, output)
    X_not = np.array([[0], [1]])
    y_not = np.array([1, 0])
    w, b = train(X_not, y_not)
    output = predict(X_not, w, b)
    print(f"NOT   actual={y_not.tolist()} predicted={output.tolist()} w={w} b={b:.2f}")
    assert np.array_equal(y_not, output)


def exp03_xor():
    banner("EXP 03-C: XOR USING TRAINED MULTI-LAYER PERCEPTRON")
    from sklearn.neural_network import MLPClassifier
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    model = MLPClassifier(hidden_layer_sizes=(4,), activation="tanh",
                          solver="lbfgs", max_iter=10000, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)
    print("Input | Actual | Predicted")
    for xi, actual, predicted in zip(X, y, predictions):
        print(xi, "|", actual, "|", predicted)
    print("Accuracy:", np.mean(predictions == y) * 100, "%")
    assert np.array_equal(predictions, y)


def exp03_all():
    exp03_activation()
    exp03_perceptron()
    exp03_xor()


# ============================================================================
# EXPERIMENT 05 — BACKPROPAGATION: ONE NUMERICAL WEIGHT-UPDATE STEP
# Exact initial weights and input from reference PDF (binary sigmoid).
# This example has 2 inputs -> 2 hidden neurons -> 1 output.
# ============================================================================
def exp05_backprop():
    banner("EXP 05: BACKPROPAGATION — HANDOUT NUMERICAL EXAMPLE")
    x = np.array([0.0, 1.0])
    target = 1.0
    learning_rate = 0.25
    # Rows=input nodes (x1, x2), columns=hidden neurons (z1, z2).
    V = np.array([[0.6, -0.3], [-0.1, 0.4]])
    hidden_bias = np.array([0.3, 0.5])
    W = np.array([0.4, 0.1])
    output_bias = -0.2

    def sigmoid(a):
        return 1 / (1 + np.exp(-a))

    # Forward pass
    hidden_net = x @ V + hidden_bias  # [0.2, 0.9]
    hidden = sigmoid(hidden_net)
    output_net = hidden @ W + output_bias
    output = float(sigmoid(output_net))
    print("Hidden nets:", np.round(hidden_net, 6))
    print("Hidden outputs:", np.round(hidden, 6))
    print("Output before update:", round(output, 6))

    # Backward pass: calculate all deltas BEFORE changing weights.
    output_delta = (target - output) * output * (1 - output)
    hidden_delta = hidden * (1 - hidden) * W * output_delta
    # Gradient updates
    new_W = W + learning_rate * output_delta * hidden
    new_output_bias = output_bias + learning_rate * output_delta
    new_V = V + learning_rate * np.outer(x, hidden_delta)
    new_hidden_bias = hidden_bias + learning_rate * hidden_delta

    print("Output delta:", round(float(output_delta), 6))
    print("Hidden deltas:", np.round(hidden_delta, 6))
    print("New input-to-hidden weights (rows=x1,x2):\n", np.round(new_V, 6))
    print("New hidden biases:", np.round(new_hidden_bias, 6))
    print("New hidden-to-output weights:", np.round(new_W, 6))
    print("New output bias:", round(float(new_output_bias), 6))
    return new_V, new_hidden_bias, new_W, new_output_bias


# ============================================================================
# EXPERIMENT 06-A — KOHONEN SELF-ORGANIZING MAP (SOM)
# Dummy clinical-like vectors: NOT the Heart Failure Clinical Records Dataset.
# The PDF gives a case-study outline, not implementation code.
# ============================================================================
def exp06_som():
    banner("EXP 06-A: SOM — DEMO WITH SYNTHETIC DATA (NO PATIENT CSV PROVIDED)")
    rng = np.random.default_rng(7)
    # Columns: age, ejection_fraction, serum_creatinine (illustrative only).
    X = np.vstack([
        rng.normal([45, 55, 0.9], [6, 5, 0.15], size=(30, 3)),
        rng.normal([70, 28, 2.0], [5, 5, 0.3], size=(30, 3)),
    ])
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    side = 4
    weights = rng.normal(size=(side, side, X.shape[1]))
    total_steps = 400
    for t in range(total_steps):
        xi = X[t % len(X)]
        distances = np.linalg.norm(weights - xi, axis=2)
        win_i, win_j = np.unravel_index(np.argmin(distances), distances.shape)
        rate = 0.5 * (0.05 / 0.5) ** (t / total_steps)
        radius = max(0.5, 2.0 * (0.5 / 2.0) ** (t / total_steps))
        for i in range(side):
            for j in range(side):
                grid_distance_sq = (i - win_i) ** 2 + (j - win_j) ** 2
                influence = np.exp(-grid_distance_sq / (2 * radius**2))
                weights[i, j] += rate * influence * (xi - weights[i, j])
    bmus = [np.unravel_index(np.argmin(np.linalg.norm(weights - xi, axis=2)),
                             (side, side)) for xi in X]
    counts = np.zeros((side, side), dtype=int)
    for i, j in bmus:
        counts[i, j] += 1
    print("SOM grid occupancy (number of synthetic records in each neuron):")
    print(counts)
    print("Total assigned:", int(counts.sum()))


# ============================================================================
# EXPERIMENT 06-B — LEARNING VECTOR QUANTIZATION (LVQ)
# Uses sklearn's offline handwritten DIGITS (8x8), as in example code.
# ============================================================================
def exp06_lvq():
    banner("EXP 06-B: LVQ — HANDWRITTEN DIGIT CLASSIFICATION")
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    digits = load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    rng = np.random.default_rng(42)
    prototypes = {}
    labels = {}
    for c in np.unique(y_train):
        mean = np.mean(X_train[y_train == c], axis=0)
        for p in range(3):
            key = (int(c), p)
            prototypes[key] = mean + 0.01 * rng.standard_normal(X_train.shape[1])
            labels[key] = int(c)
    lr = 0.01
    for epoch in range(50):
        indices = rng.permutation(len(X_train))
        for idx in indices:
            xi, yi = X_train[idx], y_train[idx]
            winner = min(prototypes, key=lambda k: np.linalg.norm(xi - prototypes[k]))
            sign = 1 if labels[winner] == yi else -1
            prototypes[winner] += sign * lr * (xi - prototypes[winner])
    predictions = []
    for xi in X_test:
        winner = min(prototypes, key=lambda k: np.linalg.norm(xi - prototypes[k]))
        predictions.append(labels[winner])
    print(f"Test accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print("First 10 actual:   ", y_test[:10])
    print("First 10 predicted:", predictions[:10])


def exp06_all():
    exp06_som()
    exp06_lvq()


# ============================================================================
# EXPERIMENT 07-A — HOPFIELD ASSOCIATIVE MEMORY
# Illustrative 5x5 digit patterns, since MNIST itself is not uploaded.
# ============================================================================
def exp07_hopfield():
    banner("EXP 07-A: HOPFIELD NETWORK — NOISY DIGIT RECALL")
    # +1 = stroke and -1 = background. A small 5x5 digit '0'.
    rows = [
        "01110",
        "10001",
        "10001",
        "10001",
        "01110",
    ]
    original = np.array([1 if ch == "1" else -1 for row in rows for ch in row])
    # Hebbian auto-associative learning: outer product, no self-weights.
    W = np.outer(original, original) / len(original)
    np.fill_diagonal(W, 0)
    noisy = original.copy()
    noisy[[0, 3, 13]] *= -1  # deliberately flip 3 pixels
    recalled = noisy.copy()
    # Asynchronous deterministic updates, preserving old state on tie.
    for _ in range(30):
        before = recalled.copy()
        for i in range(len(recalled)):
            net = W[i] @ recalled
            if net > 0:
                recalled[i] = 1
            elif net < 0:
                recalled[i] = -1
        if np.array_equal(before, recalled):
            break

    def show(name, pattern):
        print(name)
        a = pattern.reshape(5, 5)
        for row in a:
            print("".join("#" if v == 1 else "." for v in row))

    show("Stored pattern:", original)
    show("Noisy pattern:", noisy)
    show("Recalled pattern:", recalled)
    print("Pixel accuracy:", round(100 * np.mean(recalled == original), 2), "%")


# ============================================================================
# EXPERIMENT 07-B — BIDIRECTIONAL ASSOCIATIVE MEMORY (BAM)
# Bipolar one-hot English/Hindi encoding, Hebbian outer products.
# ============================================================================
def exp07_bam():
    banner("EXP 07-B: BAM — ENGLISH <-> HINDI ASSOCIATIONS")
    english = ["Dog", "Cat", "Sun"]
    hindi = ["कुत्ता", "बिल्ली", "सूरज"]
    n = len(english)
    # Orthogonal bipolar codes (diagonal +1, off diagonal -1).
    codes = np.full((n, n), -1.0)
    np.fill_diagonal(codes, 1.0)
    # Same code layout for both vocabularies, attached to different words.
    X, Y = codes.copy(), codes.copy()
    W = np.zeros((n, n))
    for xi, yi in zip(X, Y):
        W += np.outer(xi, yi)  # Hebbian learning

    def nearest_code(vectors, test):
        distances = np.sum(vectors != test, axis=1)
        return int(np.argmin(distances))

    for i in range(n):
        forward = np.where(X[i] @ W >= 0, 1, -1)
        reverse = np.where(Y[i] @ W.T >= 0, 1, -1)
        hi = nearest_code(Y, forward)
        en = nearest_code(X, reverse)
        print(f"{english[i]} -> {hindi[hi]} | {hindi[i]} -> {english[en]}")
    print("Note: illustrative encoded word pairs, not machine translation.")


def exp07_all():
    exp07_hopfield()
    exp07_bam()


# ============================================================================
# EXPERIMENT 08-A — FUZZY SETS / OPERATIONS (HANDOUT NOISE EXAMPLE)
# ============================================================================
def exp08_fuzzy_sets():
    banner("EXP 08-A: FUZZY SETS AND OPERATIONS (NO GRAPHS)")
    x = np.linspace(0, 100, 201)
    low = np.exp(-((x - 20) ** 2) / (2 * 10**2))
    high = np.exp(-((x - 80) ** 2) / (2 * 10**2))
    algebraic_sum = low + high - low * high
    algebraic_product = low * high
    bounded_sum = np.minimum(low + high, 1)
    bounded_difference = np.maximum(low - high, 0)
    for level in [20, 50, 80]:
        i = int(np.argmin(np.abs(x - level)))
        print(f"Noise={level} dB: low={low[i]:.3f}, high={high[i]:.3f}, "
              f"algebraic_sum={algebraic_sum[i]:.3f}, "
              f"algebraic_product={algebraic_product[i]:.3f}, "
              f"bounded_sum={bounded_sum[i]:.3f}, "
              f"bounded_difference={bounded_difference[i]:.3f}")


# ============================================================================
# EXPERIMENT 08-B — FUZZY RELATIONS: CARTESIAN PRODUCT AND COMPOSITIONS
# Illustrative memberships for AQI (G,M,P,S) and health (H,MS,SS).
# ============================================================================
def exp08_relations():
    banner("EXP 08-B: AQI FUZZY RELATIONS AND MAX-MIN / MAX-PRODUCT")
    # Sample fuzzy sets, not specified numerically by the PDF.
    A = np.array([0.9, 0.6, 0.3, 0.1])  # Good, Moderate, Poor, Severe
    B = np.array([0.8, 0.5, 0.2])       # Healthy, Mild, Serious
    C = np.array([0.7, 0.4, 0.3])       # Desired health condition
    # Cartesian fuzzy product R = A x B, using min t-norm.
    R = np.minimum.outer(A, B)         # shape (4 AQI, 3 health)
    # Health -> AQI relation needs transpose, so C (3) can compose with R.T (3x4).
    # Gives fuzzy membership in four AQI categories.
    max_min = np.max(np.minimum(C[:, None], R.T), axis=0)
    max_product = np.max(C[:, None] * R.T, axis=0)
    print("A (AQI G/M/P/S):", A)
    print("B (Health H/MS/SS):", B)
    print("R = A x B (rows AQI, columns health):\n", R)
    print("C (Health condition):", C)
    print("C o R^T, max-min => AQI:", max_min)
    print("C o R^T, max-product => AQI:", max_product)


def exp08_all():
    exp08_fuzzy_sets()
    exp08_relations()


# ============================================================================
# EXPERIMENT 09 — MAMDANI FUZZY LOGIC CONTROLLERS
# Pure-NumPy Mamdani engine: fuzzify, rule min, aggregate max, centroid.
# Reconstructs the handout's fan example and traffic-light case study.
# ============================================================================
def triangle(x, a, b, c):
    """Triangular MF; supports shoulder triangles a=b or b=c."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x, dtype=float)
    if a < b:
        y = np.maximum(y, np.clip((x - a) / (b - a), 0, 1))
    else:
        y = np.where(x <= b, 1.0, y)
    if b < c:
        y = np.minimum(y, np.clip((c - x) / (c - b), 0, 1)) if a < b else np.where(x <= b, 1.0, np.clip((c - x) / (c - b), 0, 1))
    else:
        y = np.where(x >= b, 1.0, y)
    return np.clip(y, 0, 1)


def trapezoid(x, a, b, c, d):
    """Trapezoidal MF with possible zero-width shoulder ramps."""
    x = np.asarray(x, dtype=float)
    left = np.clip((x - a) / (b - a), 0, 1) if b > a else np.where(x >= a, 1.0, 0.0)
    right = np.clip((d - x) / (d - c), 0, 1) if d > c else np.where(x <= d, 1.0, 0.0)
    return np.minimum(left, right)


def centroid(universe, membership):
    """Discrete centroid (weighted average on a dense/uniform output grid)."""
    denominator = np.sum(membership)
    if denominator <= 0:
        raise ValueError("No fuzzy rule fired; cannot defuzzify")
    return float(np.sum(universe * membership) / denominator)


def exp09_fan():
    banner("EXP 09-A: MAMDANI FAN SPEED — TEMPERATURE CONTROLLER")
    temp = 29.0  # input value omitted in source's example; illustrative test
    temperature_mfs = {
        "low": triangle(temp, 0, 0, 20),
        "medium": triangle(temp, 10, 20, 30),
        "high": triangle(temp, 25, 40, 40),
    }
    universe = np.arange(0, 101, 1, dtype=float)
    speed_mfs = {
        "slow": triangle(universe, 0, 0, 50),
        "medium": triangle(universe, 25, 50, 75),
        "fast": triangle(universe, 60, 100, 100),
    }
    rules = [("low", "slow"), ("medium", "medium"), ("high", "fast")]
    combined = np.zeros_like(universe)
    for condition, response in rules:
        strength = temperature_mfs[condition]
        combined = np.maximum(combined, np.minimum(strength, speed_mfs[response]))
    print("Temperature:", temp)
    print("Fuzzified memberships:", {k: float(v) for k, v in temperature_mfs.items()})
    print(f"Recommended fan speed: {centroid(universe, combined):.2f}%")


def exp09_traffic():
    banner("EXP 09-B: MAMDANI TRAFFIC GREEN-LIGHT CONTROLLER")
    density = 35.0    # vehicles/min (example in PDF)
    waiting = 100.0   # seconds (example in PDF)
    density_mfs = {
        "low": trapezoid(density, 0, 0, 8, 15),
        "medium": triangle(density, 10, 20, 30),
        "high": trapezoid(density, 25, 35, 50, 50),
    }
    waiting_mfs = {
        "short": trapezoid(waiting, 0, 0, 15, 30),
        "moderate": triangle(waiting, 20, 50, 80),
        "long": trapezoid(waiting, 70, 90, 120, 120),
    }
    seconds = np.linspace(10, 90, 801)
    duration = {
        "short": triangle(seconds, 10, 10, 30),
        "medium": triangle(seconds, 30, 45, 60),
        "long": triangle(seconds, 60, 90, 90),
    }
    # Illustrative complete nine-rule base (source gives only examples).
    rule_base = {
        ("low", "short"): "short", ("low", "moderate"): "short", ("low", "long"): "medium",
        ("medium", "short"): "medium", ("medium", "moderate"): "medium", ("medium", "long"): "long",
        ("high", "short"): "medium", ("high", "moderate"): "long", ("high", "long"): "long",
    }
    aggregated = np.zeros_like(seconds)
    for (density_term, wait_term), output_term in rule_base.items():
        strength = min(float(density_mfs[density_term]), float(waiting_mfs[wait_term]))
        aggregated = np.maximum(aggregated, np.minimum(strength, duration[output_term]))
    print("Density memberships:", {k: round(float(v), 3) for k, v in density_mfs.items()})
    print("Waiting memberships:", {k: round(float(v), 3) for k, v in waiting_mfs.items()})
    print(f"Recommended green duration: {centroid(seconds, aggregated):.2f} seconds")
    print("Note: membership parameters/rule base are illustrative choices within the ranges in the handout.")


def exp09_all():
    exp09_fan()
    exp09_traffic()


# ============================================================================
# EXPERIMENT 10 — ANFIS: FIRST-ORDER SUGENO NEURO-FUZZY SYSTEM
# Educational implementation: two inputs, 2 MFs each -> 4 Sugeno rules.
# Training: least squares for consequents, optimizes Gaussian MF premise
# centres (alternating hybrid update). Synthetic energy-use-like data.
# ============================================================================
def exp10_anfis():
    banner("EXP 10: ANFIS — HYBRID LEARNING, SYNTHETIC ENERGY EXAMPLE")
    from scipy.optimize import minimize
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    rng = np.random.default_rng(12)
    # Synthetic simplified case: temperature (C) and occupancy (people).
    # The PDF also proposes humidity, appliances and time; those are not
    # included in this minimal 2-feature teaching implementation.
    X = np.column_stack([rng.uniform(18, 38, 180), rng.uniform(0, 5, 180)])
    y = (0.4 * X[:, 0] + 1.7 * X[:, 1] +
         2.0 * np.maximum(X[:, 0] - 28, 0) +
         rng.normal(0, 0.5, len(X)))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)
    xmean, xstd = X_train.mean(axis=0), X_train.std(axis=0)
    train = (X_train - xmean) / xstd
    test = (X_test - xmean) / xstd
    # Initial Gaussian centers: low and high for each input.
    centers = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    widths = np.ones_like(centers) * 0.85

    def design_matrix(inputs, flat_centers):
        c = flat_centers.reshape(2, 2)
        mu = np.exp(-0.5 * ((inputs[:, :, None] - c[None, :, :]) /
                             widths[None, :, :]) ** 2)  # N x 2 variables x 2 MFs
        # Rule firing strengths for combinations (low/low, low/high, ...).
        firing = np.stack([mu[:, 0, i] * mu[:, 1, j]
                           for i in range(2) for j in range(2)], axis=1)
        normalized = firing / (firing.sum(axis=1, keepdims=True) + 1e-12)
        # Sugeno consequents f_r = p_r*x1 + q_r*x2 + r_r.
        augmented = np.column_stack([inputs, np.ones(len(inputs))])
        return (normalized[:, :, None] * augmented[:, None, :]).reshape(len(inputs), -1)

    def solve_consequents(flat_centers):
        A = design_matrix(train, flat_centers)
        coeffs = np.linalg.lstsq(A, y_train, rcond=None)[0]
        return coeffs

    def objective(flat_centers):
        coeffs = solve_consequents(flat_centers)  # LSE step
        pred = design_matrix(train, flat_centers) @ coeffs
        mse = np.mean((pred - y_train) ** 2)
        return mse + 1e-5 * np.sum(flat_centers**2)

    # Alternating hybrid optimization: premise centers are optimized via
    # finite-difference gradient estimates (L-BFGS-B), while each objective
    # evaluation refits the linear consequent parameters by least squares.
    bounds = [(-2.5, 2.5)] * 4
    result = minimize(objective, centers.ravel(), method="L-BFGS-B",
                      bounds=bounds, options={"maxiter": 80, "ftol": 1e-9})
    trained_centers = result.x
    consequent_weights = solve_consequents(trained_centers)
    predicted = design_matrix(test, trained_centers) @ consequent_weights
    print("Learned Gaussian centers:\n", trained_centers.reshape(2, 2))
    print("Number of Sugeno rules:", 4)
    print("RMSE:", round(float(np.sqrt(mean_squared_error(y_test, predicted))), 3))
    print("MAE:", round(float(mean_absolute_error(y_test, predicted)), 3))
    print("R2:", round(float(r2_score(y_test, predicted)), 3))
    print("First 5 actual / predicted energy values:")
    for actual, pred in zip(y_test[:5], predicted[:5]):
        print(f"  {actual:.2f} / {pred:.2f}")
    print("Note: 2-feature ANFIS demonstration, not the full 4-5-input case study.")


# ============================================================================
# COMMAND-LINE MENU — RUN ONE SECTION OR ALL
# ============================================================================
SECTIONS = {
    "1": exp01_missionaries,
    "1.2": exp012_water_jug,
    "2": exp02_wumpus,
    "3": exp03_all,
    "3-activation": exp03_activation,
    "3-perceptron": exp03_perceptron,
    "3-xor": exp03_xor,
    "5": exp05_backprop,
    "6": exp06_all,
    "6-som": exp06_som,
    "6-lvq": exp06_lvq,
    "7": exp07_all,
    "7-hopfield": exp07_hopfield,
    "7-bam": exp07_bam,
    "8": exp08_all,
    "8-sets": exp08_fuzzy_sets,
    "8-relations": exp08_relations,
    "9": exp09_all,
    "9-fan": exp09_fan,
    "9-traffic": exp09_traffic,
    "10": exp10_anfis,
}


def main():
    if len(sys.argv) == 1 or sys.argv[1] in ("all", "--all"):
        # Run full experiment groups once each; do not duplicate sub-sections.
        for key in ["1", "1.2", "2", "3", "5", "6", "7", "8", "9", "10"]:
            SECTIONS[key]()
    elif sys.argv[1] in ("--list", "-l"):
        print("Available experiment identifiers:")
        print(", ".join(SECTIONS))
        print("Experiment 4 was not supplied.")
    elif sys.argv[1] in SECTIONS:
        SECTIONS[sys.argv[1]]()
    else:
        print("Unknown section:", sys.argv[1])
        print("Use --list to see all available sections.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
