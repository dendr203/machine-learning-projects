# 2048 AI Agents: Experiment Report

> **Note on AI assistance.** Much of the agent code in this project was written with the help of an AI assistant. I used it to survey candidate algorithms for 2048 and to generate the implementations of the greedy, heuristic and expectimax agents. I then ran the experiments, benchmarked every version over 30 games, analysed the results and chose the next step. Where an idea or a piece of code came from the AI, the text below says so.

My first step was to find out which algorithms are commonly used to play 2048. I asked an AI assistant to list possible approaches and describe their strengths and weaknesses. Based on its answer, I decided to start with a simple, fast algorithm and work up from there.

## Agents

### Greedy agent

The first agent the AI suggested is a greedy algorithm. On every turn it simulates a move in each of the four directions and picks the one with the highest immediate score gain, breaking ties by the number of empty cells. It behaves much like a person playing the game for the first time: it never plans ahead and only looks for the best move right now.

```python
def greedy_move(grid, score):
    moves = ['left', 'right', 'up', 'down']
    best_move = None
    best_score = -1
    best_empty = -1
    original_score = score

    for move in moves:
        # Copy the grid and score for simulation
        new_grid = cp.deepcopy(grid)
        new_score = cp.copy(score)

        # Simulate the move
        if move == 'left':
            new_grid, new_score = move_left(new_grid, new_score)
        elif move == 'right':
            new_grid, new_score = move_right(new_grid, new_score)
        elif move == 'up':
            new_grid, new_score = move_up(new_grid, new_score)
        elif move == 'down':
            new_grid, new_score = move_down(new_grid, new_score)

        # Skip moves that do not change the grid
        if np.array_equal(new_grid, grid):
            continue

        # Heuristic: score gain, then number of empty cells
        score_gain = new_score - original_score
        empty_cells = np.count_nonzero(new_grid == 0)

        # Keep the best move so far
        if (score_gain > best_score) or (score_gain == best_score and empty_cells > best_empty):
            best_score = score_gain
            best_empty = empty_cells
            best_move = move

    return best_move if best_move is not None else np.random.choice(moves)
```

**Results (30 games):**

```
best_score: 7084
worst_score: 1024
average_score: 3335.6
wins: 0
losses: 30
average_max_tile: 260.26666666666665
average_moves: 273.76666666666665
```

Unsurprisingly, such a simple algorithm does not perform well. I expected more, but it is still interesting that an agent that never looks ahead can reach a fairly high total score. The total score, however, does not decide the game: all that matters is reaching the 2048 tile. On average, this agent does not even get past a maximum tile of 300.

### Weighted heuristic agent

The next idea from the AI was to rely purely on heuristics: the agent picks moves based only on a set of predefined rules, which should also keep it fast. If I already have rules that describe good play, the agent should do very well, right? The opposite turned out to be true. Even though each individual rule is sensible, following them greedily does not necessarily lead to a good overall game.

Unlike the greedy agent, which optimises a single criterion (immediate score gain, with empty cells as a tie-breaker), this agent combines several criteria:

* `score_gain`: maximise the immediate score gain.
* `empty_cells`: keep as many empty cells as possible, which lowers the chance of the game ending.
* `calculate_smoothness`: penalise large differences between neighbouring cells, so that equal tiles stay close together and can be merged quickly.
* `calculate_monotonicity`: prefer values ordered in ascending or descending order along rows and columns, with high values in a corner.
* `count_available_moves`: the number of moves currently available.
* `edge_bonus`: as with monotonicity, keep high values in a corner so new small tiles can be merged quickly without scrambling the board.
* `max_tile_corner_bonus`: an extra reward for keeping the largest tile in a corner (defined in the code, but not included in the final weighted sum).

The heuristic value of each move is a weighted sum of these features; the weights are shown in the code below.

```python
def calculate_smoothness(grid):
    """Penalize large differences between neighbouring tiles (log2 scale)."""
    smoothness = 0
    log_grid = np.log2(grid + (grid == 0))  # avoid log(0)

    for i in range(4):
        for j in range(4):
            if i < 3:
                smoothness -= (log_grid[i, j] - log_grid[i+1, j]) ** 2  # squared penalty
            if j < 3:
                smoothness -= (log_grid[i, j] - log_grid[i, j+1]) ** 2

    return smoothness


def calculate_monotonicity(grid):
    """Reward boards whose values decrease monotonically along rows and columns."""
    monotonicity = 0
    log_grid = np.log2(grid + (grid == 0))

    for i in range(4):
        for j in range(3):
            if log_grid[i, j] > log_grid[i, j+1]:
                monotonicity += log_grid[i, j] - log_grid[i, j+1]

    for j in range(4):
        for i in range(3):
            if log_grid[i, j] > log_grid[i+1, j]:
                monotonicity += log_grid[i, j] - log_grid[i+1, j]

    return monotonicity


def edge_bonus(grid):
    """Positional bonus for large tiles near the top-left corner."""
    edge_weights = np.array([
        [6, 5, 4, 3],
        [5, 4, 3, 2],
        [4, 3, 2, 1],
        [3, 2, 1, 0]
    ])

    return np.sum(grid * edge_weights)

def count_available_moves(grid):
    """Number of legal moves (higher is better)."""
    count = 0
    for move in ['left', 'right', 'up', 'down']:
        new_grid = cp.deepcopy(grid)
        new_score = 0  # score is irrelevant here; we only check whether the move changes the grid

        if move == 'left':
            new_grid, _ = move_left(new_grid, new_score)
        elif move == 'right':
            new_grid, _ = move_right(new_grid, new_score)
        elif move == 'up':
            new_grid, _ = move_up(new_grid, new_score)
        elif move == 'down':
            new_grid, _ = move_down(new_grid, new_score)

        if not np.array_equal(new_grid, grid):
            count += 1

    return count


def max_tile_corner_bonus(grid):
    """Return a bonus if the largest tile sits in one of the corners (defined but not used by heuristic_move)."""
    max_tile = np.max(grid)
    if grid[0, 0] == max_tile or grid[0, 3] == max_tile or grid[3, 0] == max_tile or grid[3, 3] == max_tile:
        return 100  # corner-strategy bonus
    return 0



def heuristic_move(grid, score):
    moves = ['left', 'right', 'up', 'down']
    best_move = None
    best_heuristic = -float('inf')
    original_score = score

    for move in moves:
        new_grid = cp.deepcopy(grid)
        new_score = cp.copy(score)

        # Simulate the move
        if move == 'left':
            new_grid, new_score = move_left(new_grid, new_score)
        elif move == 'right':
            new_grid, new_score = move_right(new_grid, new_score)
        elif move == 'up':
            new_grid, new_score = move_up(new_grid, new_score)
        elif move == 'down':
            new_grid, new_score = move_down(new_grid, new_score)

        if np.array_equal(new_grid, grid):
            continue  # move did not change the grid; ignore it

        # Compute heuristic features
        score_gain = new_score - original_score
        empty_cells = np.count_nonzero(new_grid == 0)
        smoothness = calculate_smoothness(new_grid)
        monotonicity = calculate_monotonicity(new_grid)
        move_count = count_available_moves(new_grid)
        corner_bonus = edge_bonus(new_grid)

        # Weighted combination of the features
        heuristic = (
            1.0 * score_gain +
            2.0 * empty_cells +
            3.0 * smoothness +
            2.0 * monotonicity +
            1.5 * move_count +
            3.0 * corner_bonus
        )

        if heuristic > best_heuristic:
            best_heuristic = heuristic
            best_move = move

    return best_move if best_move is not None else np.random.choice(moves)
```

Because the agent is "rewarded" for good board arrangements, it might look like reinforcement learning. It is not: the agent does not take past moves into account and learns nothing from them. All computations and rewards concern only the current move.

**Results (30 games):**

```
best_score: 6640
worst_score: 740
average_score: 2784.9333333333334
wins: 0
losses: 30
average_max_tile: 211.2
average_moves: 243.63333333333333
```

Score gain is no longer the agent's only goal, yet the results are still poor (slightly worse than the greedy agent). To win, the agent needs to look ahead at future moves.

### Expectimax

After two unsuccessful approaches, I searched online and found on several forums that expectimax works very well for 2048. The reason is that there is no opponent trying to minimise our score (which would call for minimax); the game only places new tiles at random. The agent therefore plans its moves ahead so that it does not block the grid and can eventually win. I asked the AI to implement this algorithm.

```python
def evaluate_grid(grid):
    """Heuristic evaluation of a grid."""
    return np.sum(grid) + 10 * np.count_nonzero(grid == 0)  # simple heuristic: tile sum + empty-cell bonus

def expectimax(grid, depth, player_turn):
    """Depth-limited expectimax search for 2048."""

    if depth == 0:
        return evaluate_grid(grid), None  # leaf: return heuristic value, no move

    if player_turn:
        # Max node: the player picks the best move
        best_move = None
        best_value = -float('inf')

        for move in ['left', 'right', 'up', 'down']:
            new_grid = cp.deepcopy(grid)
            new_score = 0

            if move == 'left':
                new_grid, new_score = move_left(new_grid, new_score)
            elif move == 'right':
                new_grid, new_score = move_right(new_grid, new_score)
            elif move == 'up':
                new_grid, new_score = move_up(new_grid, new_score)
            elif move == 'down':
                new_grid, new_score = move_down(new_grid, new_score)

            if np.array_equal(new_grid, grid):
                continue  # move did not change the grid

            value, _ = expectimax(new_grid, depth - 1, player_turn=False)

            if value > best_value:
                best_value = value
                best_move = move

        return best_value, best_move

    else:
        # Chance node: the game spawns a 2 or a 4
        empty_cells = np.argwhere(grid == 0)
        if len(empty_cells) == 0:
            return evaluate_grid(grid), None  # no empty cells

        total_expectation = 0
        for (x, y) in empty_cells:
            test_grid = grid.copy()

            # Spawn a 2 (probability 0.9)
            test_grid[x, y] = 2
            value_2, _ = expectimax(test_grid, depth - 1, player_turn=True)

            # Spawn a 4 (probability 0.1)
            test_grid[x, y] = 4
            value_4, _ = expectimax(test_grid, depth - 1, player_turn=True)

            total_expectation += 0.9 * value_2 + 0.1 * value_4

        return total_expectation / len(empty_cells), None
```

The algorithm alternates between the agent's (player's) move and a walk through the probability tree of random tile placements. It explores the possible outcomes down to a fixed depth and then picks the move with the best expected value.

The search starts with the player's turn. For every possible move, expectimax calls itself recursively with the depth reduced by one, and it becomes the game's turn, which is simulated by placing a 2 or a 4 on every empty cell. This alternation continues until the depth limit is reached, where the leaves are scored with the heuristic function. The result is the move that is most likely to lead to the best outcome. The depth of the tree therefore controls both how well the agent plays and how fast it runs. At depth 3 the agent is fairly fast but less successful; increasing the depth to just 5 makes it several times slower. That is because two extra levels do not add just two levels of states: they add two levels of branching on top of every state that already exists.

**Results (30 games, depth 4):**

```
best_score: 15836
worst_score: 3404
average_score: 9058.666666666666
wins: 0
losses: 30
average_max_tile: 708.2666666666667
average_moves: 567.5666666666667
```

All metrics improved substantially compared with the two previous agents, but there is still no win. The weak point may be the simple heuristic function, so the next version uses a more complex one. To speed the agent up, the search depth is also set dynamically: early in the game (moves 0 to 200) a deep search is not needed, while in later stages (600 moves and more) a deeper search is necessary to choose good moves.

```python
def evaluate_grid(grid):
    """Improved heuristic evaluation of a grid."""
    # Sum of tile values
    sum_value = np.sum(grid)

    # Number of empty cells
    empty_cells = np.count_nonzero(grid == 0)

    # Bonus for large tiles in the corners
    corner_bonus = 0
    corner_bonus += grid[0, 0] * 0.1
    corner_bonus += grid[0, 3] * 0.1
    corner_bonus += grid[3, 0] * 0.1
    corner_bonus += grid[3, 3] * 0.1

    # Smoothness: number of equal adjacent tile pairs (possible merges)
    smoothness = 0
    for i in range(4):
        for j in range(4):
            if i < 3 and grid[i, j] == grid[i + 1, j]:
                smoothness += 1
            if j < 3 and grid[i, j] == grid[i, j + 1]:
                smoothness += 1

    # Weighted combination
    return sum_value + 10 * empty_cells + corner_bonus + 10 * smoothness
```

```python
        # Increase the search depth as the game progresses
        if i < 200:
            depth = 2
        elif i < 400:
            depth = 3
        elif i < 600:
            depth = 4
        else:
            depth = 5
```

**Results (30 games, dynamic depth):**

```
best_score: 16720
worst_score: 5604
average_score: 9678.758620689656
wins: 0
losses: 29
average_max_tile: 688.551724137931
average_moves: 612.3793103448276
```

Only 29 of the 30 games are included: one game reached the 1000-move cap without ending, so it was not recorded.

We are getting closer to a win, but this version of expectimax still has none. Move selection needs further improvement. The AI suggested improving the evaluation of the board itself, similar to the pure heuristic agent: the evaluation combines several features that together determine which move is best.

```python
def evaluate_board(grid):
    empty_tiles = np.sum(grid == 0)
    max_tile = np.max(grid)
    monotonicity = calculate_monotonicity(grid)
    smoothness = calculate_smoothness(grid)

    # Weighted score
    return (
        2 * empty_tiles +
        4 * max_tile +
        1 * monotonicity +
        1 * smoothness
    )

def calculate_monotonicity(grid):
    score = 0
    for row in grid:
        for i in range(3):
            if row[i] > row[i + 1]:
                score -= abs(row[i] - row[i + 1])
    for col in grid.T:
        for i in range(3):
            if col[i] > col[i + 1]:
                score -= abs(col[i] - col[i + 1])
    return score

def calculate_smoothness(grid):
    score = 0
    for row in grid:
        for i in range(3):
            if row[i] == row[i + 1]:
                score += row[i]
    for col in grid.T:
        for i in range(3):
            if col[i] == col[i + 1]:
                score += col[i]
    return score
```

**Results (30 games, depth 3 after the root move):**

```
best_score: 17560
worst_score: 3440
average_score: 13022.37037037037
wins: 11
losses: 16
average_max_tile: 862.8148148148148
average_moves: 764.5555555555555
```

Finally, some wins. A good heuristic function is key to expectimax's decisions, because it determines how the quality of each grid state is judged.

> **Caveats on these numbers.** Only 27 of the 30 games are included in the statistics: three games reached the 1000-move cap without a win or game over and were not recorded. In addition, this agent runs its lookahead through `play_2048`, which raises the `WIN` signal as soon as a *simulated* move creates a 2048 tile. A "win" is therefore recorded when the search reaches 2048, which can be a move or two before the tile appears on the real board. In the saved demonstration game, for example, the game ends with a "WIN" while the largest tile on the board is 1024 (next to two adjacent 512 tiles that can still merge).

## Reflections

A lot of things can be labelled "AI" as long as they involve some kind of decision-making based on an input. As I see it, an AI system can choose which of many goals to pursue, whereas an algorithm has a single predefined goal to achieve. This is why nearly every company today wants, or claims to already have, "AI", even when it is not true AI but, for example, a decision tree.

Because companies now present basic algorithms and decision logic under the AI label, one could say that AI is already used almost everywhere. I am not against using "AI" in, for example, household appliances where it makes them easier to use, but I would not call it AI.

In my view, true AI will exist only when it can think like a human, in the sense of connecting data that is only loosely related, or not related at all. People can find connections even where there may be none. And above all, although this is probably not achievable for general AI, it would need some kind of personality, which is what separates us from machines.

One promising application of machine learning comes from my current job. The company builds and tests many embedded devices that communicate over a network, so both the servers and the devices produce a large volume of logs that are sent to a central server. The logs come in many formats (XML, JSON, connectivity status, network dumps, and so on). We have years of historical logs and, importantly, logs where we know a problem or anomaly occurred. We would like to train a model that can at least detect anomalies in incoming logs, so that an engineer only needs to review the flagged section instead of digging through a log with hundreds of thousands of lines.
