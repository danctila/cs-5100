"""
8-puzzle problem.

The 8-puzzle is one of the canonical examples used in the course (mod3.6)
to compare heuristic quality.  The mod3.6 table directly compares expansion
counts for A*(h1) and A*(h2) across solution depths, which our experiments
reproduce.

State representation
--------------------
A flat tuple of 9 integers.  Position i in the tuple corresponds to cell
(i // 3, i % 3) in the 3x3 grid.  The blank tile is represented as 0.

    (1, 2, 3,         1 2 3
     4, 5, 6,   ==>   4 5 6
     7, 8, 0)         7 8 _

Goal state: (1, 2, 3, 4, 5, 6, 7, 8, 0)

Heuristics (mod3.6 "Heuristics for the 8-Puzzle" section)
----------------------------------------------------------
h1 -- Misplaced Tiles
    Count of tiles that are not in their goal position (blank excluded).
    Admissible: every misplaced tile needs at least 1 move to reach its goal.
    Weaker heuristic.

h2 -- Manhattan Distance
    Sum over all tiles (blank excluded) of
        |row_current - row_goal| + |col_current - col_goal|
    Admissible: a tile cannot reach its goal in fewer moves than its
    Manhattan distance (each step moves it one cell).
    Dominates h1: h2(n) >= h1(n) for all n  (mod3.6).
    Stronger heuristic -> fewer node expansions under A*.
"""

import random
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GOAL: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 0)

# Precompute goal position of each tile for fast Manhattan lookup.
# GOAL_POS[tile] = (row, col) where the tile lives in the goal state.
GOAL_POS = {tile: (i // 3, i % 3) for i, tile in enumerate(GOAL)}

_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # up, down, left, right


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------

class PuzzleProblem:
    """
    8-puzzle problem compatible with search.py.

    The state is an immutable tuple so it can be stored in sets and used
    as dictionary keys (required by the explored set in best_first_search).
    """

    def __init__(self, start: Tuple[int, ...]):
        self.start = start

    # --- problem interface -------------------------------------------------

    def initial_state(self) -> Tuple[int, ...]:
        return self.start

    def goal_test(self, state: Tuple[int, ...]) -> bool:
        return state == GOAL

    def successors(self, state: Tuple[int, ...]):
        """
        Yield (next_state, step_cost=1) for each legal slide of the blank.

        The blank moves by swapping with an adjacent tile (up/down/left/right).
        Step cost is 1 for every move, so path cost equals number of moves.
        """
        blank = state.index(0)
        r, c = blank // 3, blank % 3
        for dr, dc in _MOVES:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 3 and 0 <= nc < 3:
                neighbor = nr * 3 + nc
                new_state = list(state)
                new_state[blank], new_state[neighbor] = (
                    new_state[neighbor], new_state[blank]
                )
                yield tuple(new_state), 1.0

    # --- heuristics --------------------------------------------------------

    @staticmethod
    def h1(state: Tuple[int, ...]) -> float:
        """
        Misplaced Tiles heuristic (mod3.6).

        h1(n) = number of tiles not in goal position (blank excluded).
        Admissible: each misplaced tile needs >= 1 move.
        """
        return float(sum(
            1 for i, tile in enumerate(state)
            if tile != 0 and tile != GOAL[i]
        ))

    @staticmethod
    def h2(state: Tuple[int, ...]) -> float:
        """
        Manhattan Distance heuristic (mod3.6).

        h2(n) = sum of |row_curr - row_goal| + |col_curr - col_goal|
                for each tile (blank excluded).

        Admissible and consistent. Dominates h1 (h2 >= h1 for all n),
        so A* with h2 expands no more nodes than A* with h1 (mod3.6).
        """
        total = 0.0
        for i, tile in enumerate(state):
            if tile == 0:
                continue
            r, c = i // 3, i % 3
            gr, gc = GOAL_POS[tile]
            total += abs(r - gr) + abs(c - gc)
        return total


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------

def generate_instance(
    depth: int,
    rng: random.Random,
) -> Tuple[int, ...]:
    """
    Generate a scrambled 8-puzzle by making `depth` random moves from GOAL.

    Starting from the goal and walking backward guarantees solvability.
    Immediately reversing the previous move is avoided so the walk does
    not trivially undo itself.

    The actual optimal solution depth of the returned state may be less
    than `depth` if the walk revisited states -- this is normal and
    noted in the experiment output.
    """
    state = list(GOAL)
    blank = state.index(0)
    prev_blank = -1   # index of blank before this step (avoid backtrack)

    for _ in range(depth):
        r, c = blank // 3, blank % 3
        candidates = []
        for dr, dc in _MOVES:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 3 and 0 <= nc < 3:
                nb = nr * 3 + nc
                if nb != prev_blank:   # don't immediately reverse last move
                    candidates.append(nb)

        new_blank = rng.choice(candidates)
        state[blank], state[new_blank] = state[new_blank], state[blank]
        prev_blank = blank
        blank = new_blank

    return tuple(state)


def generate_batch(
    n: int,
    depth: int,
    seed: int,
) -> List[Tuple[int, ...]]:
    """
    Return a list of `n` scrambled instances generated with a fixed seed.

    Using a fixed seed makes experiments fully reproducible.
    """
    rng = random.Random(seed)
    return [generate_instance(depth, rng) for _ in range(n)]


# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------

def print_puzzle(state: Tuple[int, ...]) -> None:
    """Print the puzzle in a 3x3 grid; blank shown as '_'."""
    for row in range(3):
        print(" ".join(
            str(state[row * 3 + col]) if state[row * 3 + col] != 0 else "_"
            for col in range(3)
        ))
    print()
