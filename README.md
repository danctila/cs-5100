# Capstone Code

FAI CS5100 Spring 2026 | A\* Reproduction Project

## Files

| File             | What it contains                                                              |
| ---------------- | ----------------------------------------------------------------------------- |
| `search.py`      | `best_first_search`, `ucs`, `astar`, `greedy` (all four algorithms)           |
| `grid.py`        | `GridProblem`, `make_grid`, `print_grid`, `manhattan` heuristic               |
| `puzzle.py`      | `PuzzleProblem`, `h1` (misplaced tiles), `h2` (Manhattan), instance generator |
| `run_tests.py`   | 13 sanity tests covering correctness, optimality, and path validity           |
| `experiments.py` | Full experiment suite across both domains; prints and saves 4 result tables   |
| `results.txt`    | Saved output from the last run of `experiments.py`                            |

## Algorithms

All four algorithms are implemented in `search.py` as thin wrappers over a single
`best_first_search` function. The only difference is the evaluation function
used to order the frontier:

```
UCS    : f(n) = g(n)           (uniform-cost search)
A*     : f(n) = g(n) + h(n)    (optimal when h is admissible)
Greedy : f(n) = h(n)           (fast but not optimal in general)
```

When `h(n) = 0`, A\* reduces to UCS exactly.

## Domains

**Grid pathfinding:** 4-neighbor unit-cost grids with random obstacles.
Heuristic: Manhattan distance (admissible and consistent).

**8-puzzle:** classic 3x3 sliding-tile problem.
Two heuristics:

- `h1`: misplaced tiles (admissible, weaker)
- `h2`: Manhattan distance of each tile from its goal position (admissible, dominates h1)

## Run the sanity tests

```bash
python run_tests.py
```

Tests check:

1. UCS and A\* both return the known optimal cost
2. A\* (Manhattan) expands ≤ nodes than UCS (heuristic dominance)
3. Obstacle routing produces a valid detour path
4. Unsolvable instances return `None`
5. Path validity on 10 random 15×15 grids (fixed seed for reproducibility)
6. Heuristic dominance on a 20×20 open grid

## Run the full experiments

```bash
python experiments.py
```

Runs all four algorithms on both domains and prints four summary tables:

1. Grid 15x15, 20 instances, seed 42
2. Grid 30x30, 20 instances, seed 42
3. 8-puzzle depth ~8, 20 instances, seed 42
4. 8-puzzle depth ~14, 20 instances, seed 42 (UCS omitted, intractable at this depth)

Each table reports average path cost, average node expansions, and suboptimal count.
Dominance checks (A\* with stronger heuristic expands ≤ nodes than A\* with weaker heuristic)
are printed after each table. Results are also saved to `results.txt`.

Expected runtime: under 1 second on a standard laptop.
