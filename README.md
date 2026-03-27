# Capstone Code — Weeks 1 & 2

FAI CS5100 Spring 2026 — A\* Reproduction Project

## Files

| File           | What it contains                                                |
| -------------- | --------------------------------------------------------------- |
| `search.py`    | `best_first_search`, `ucs`, `astar`                             |
| `grid.py`      | `GridProblem`, `make_grid`, `print_grid`, `manhattan` heuristic |
| `run_tests.py` | Sanity tests verifying Week 2 claims                            |

## Algorithms

Both algorithms are implemented in `search.py` as thin wrappers over a single
`best_first_search` function. The only difference is the evaluation function
used to order the frontier:

```
UCS : f(n) = g(n)           (mod3.5 pseudocode)
A*  : f(n) = g(n) + h(n)    (mod3.6 pseudocode)
```

When `h(n) = 0`, A\* reduces to UCS exactly.

## Run tests

```bash
python run_tests.py
```

Tests check:

1. UCS and A\* both return the known optimal cost
2. A\* (Manhattan) expands ≤ nodes than UCS (heuristic dominance, mod3.6)
3. Obstacle routing works correctly
4. Failure (no path) returns `None`
5. Path validity on 10 random 15×15 grids (fixed seed for reproducibility)
