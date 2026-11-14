# 2048 AI: Greedy, Heuristic and Expectimax Agents

A NumPy implementation of the game 2048 and a series of increasingly capable agents that play it: a one-step greedy agent, a weighted multi-heuristic agent, and several versions of depth-limited **expectimax** search. Every agent is benchmarked over 30 games, and the results show why planning under uncertainty (expectimax) beats myopic move selection, and how much the evaluation function matters.

The full write-up of the experiments, including the reasoning behind each iteration, is in **[REPORT.md](REPORT.md)**.

## Agents

All agents share the same game engine (`move_left/right/up/down`, `add_new_number`, `check_game_over`, `play_2048`). A new tile is a 2 with probability 0.9 and a 4 with probability 0.1. A game is won when a 2048 tile appears.

### 1. Greedy agent
Simulates all four moves and picks the one with the highest **immediate score gain**, breaking ties by the number of **empty cells**. No lookahead.

### 2. Weighted heuristic agent
Still one move deep, but scores each resulting board with a weighted sum of hand-crafted features:

| Feature | What it measures | Weight |
|---|---|---|
| Score gain | Points earned by the move | 1.0 |
| Empty cells | Number of free cells (more room = less risk of game over) | 2.0 |
| Smoothness | Negative sum of squared log2 differences between neighbouring tiles | 3.0 |
| Monotonicity | Sum of log2 decreases along rows and columns (rewards ordered boards) | 2.0 |
| Available moves | Number of legal moves after the move | 1.5 |
| Corner bonus | Tile values weighted by a positional matrix that peaks in the top-left corner | 3.0 |

### 3. Expectimax search
2048 is a single-player game against chance, not against an adversary, so minimax is the wrong model. Expectimax alternates two kinds of nodes:

- **Max nodes** (player): try every legal move and keep the one with the highest value.
- **Chance nodes** (game): place a 2 (p = 0.9) or a 4 (p = 0.1) on each empty cell and take the **expected value** over all spawns.

The recursion stops at a **depth limit**, where the board is scored with a heuristic evaluation function. Depth is counted in plies (each max or chance layer is one level), so depth 4 means two player moves with a tile spawn after each. The branching factor at chance nodes (up to 2 x empty cells) makes the cost grow very quickly with depth: going from depth 3 to 5 made the agent several times slower.

Three versions were tested:

| Version | Evaluation function | Depth |
|---|---|---|
| Basic expectimax | Tile sum + 10 x empty cells | Fixed, 4 |
| Dynamic-depth expectimax | Tile sum + 10 x empty cells + corner bonus + 10 x equal-neighbour pairs (the search also raises the 4-spawn probability to 0.2 once the tile sum exceeds 1000) | 2 / 3 / 4 / 5 for moves < 200 / < 400 / < 600 / later |
| Improved expectimax | 2 x empty cells + 4 x max tile + monotonicity + smoothness (raw tile values) | Root move + 3 (two player moves) |

## Results

30 games per agent, capped at 1000 moves per game. All numbers are taken from the saved notebook outputs.

| Agent | Games recorded | Wins | Average score | Best score | Average max tile | Average moves |
|---|---|---|---|---|---|---|
| Greedy | 30 | 0 | 3335.6 | 7084 | 260.3 | 273.8 |
| Weighted heuristic | 30 | 0 | 2784.9 | 6640 | 211.2 | 243.6 |
| Expectimax (depth 4) | 30 | 0 | 9058.7 | 15836 | 708.3 | 567.6 |
| Expectimax (dynamic depth) | 29 | 0 | 9678.8 | 16720 | 688.6 | 612.4 |
| **Improved expectimax** | **27** | **11** | **13022.4** | **17560** | **862.8** | **764.6** |

Notes:
- **Games recorded:** a game that hits the 1000-move cap without a win or game over is not counted, so the dynamic-depth run has 29 recorded games and the improved run has 27. The improved agent won 11 of the 30 games it played (11 wins and 16 losses among the 27 recorded games).
- **How wins are counted for the improved agent:** its lookahead uses `play_2048`, which signals a win as soon as a *simulated* move produces a 2048 tile. A win is therefore recorded when the search reaches 2048, which can be a move or two before the tile appears on the real board. See the caveat in [REPORT.md](REPORT.md).
- Averages are rounded to one decimal place; the exact values are in the notebook and the report.

**Takeaways**
- Looking ahead matters: even the simplest expectimax nearly tripled the average score and max tile of the one-step agents.
- More features do not automatically help: the weighted heuristic agent did slightly worse than plain greedy, because its hand-tuned weights are still applied only one move ahead.
- The evaluation function is the biggest lever: with the same search, switching to a monotonicity and smoothness based evaluation produced the first wins.

## How to run

Requirements: Python 3 and NumPy (plus Jupyter to open the notebooks).

```bash
pip install -r requirements.txt
pip install jupyter   # if not already installed
jupyter notebook 2048_agents.ipynb
```

Run the cells from top to bottom. The engine is defined in the first cell; each agent section has a single-game demo cell followed by a 30-game benchmark cell. The expectimax benchmarks are slow (pure Python search), so expect them to take a long time.

## File structure

```
2048-ai/
├── 2048_agents.ipynb   # Game engine, all agents and their 30-game benchmarks
├── base_game.ipynb     # Standalone 2048 game engine
├── REPORT.md           # Detailed write-up of the experiments and results
├── requirements.txt    # Python dependencies (NumPy)
└── README.md
```

## Development note

Much of the agent code was written with the help of an AI assistant, which I used to survey algorithms and generate the agent implementations. I ran the experiments, benchmarked each version and analysed the results. [REPORT.md](REPORT.md) notes where AI suggestions were used.
