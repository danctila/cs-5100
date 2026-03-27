"""
Grid pathfinding problem.

Represents the 4-neighbour unit-cost grid used throughout the capstone
experiments.  The state space matches the course description (mod3.0):

    States      : (row, col) integer pairs
    Actions     : move up / down / left / right (4-neighbour)
    Step cost   : 1  (unit cost; all edges equal)
    Obstacles   : cells marked False in the grid are impassable

Heuristics
----------
manhattan(state)
    |row_current - row_goal| + |col_current - col_goal|

    Admissible: never overestimates because every step moves at most one
    unit closer in each axis, so the true remaining cost is always >= the
    Manhattan distance.

    Consistent: satisfies the triangle inequality h(n) <= c(n,n') + h(n')
    for every edge (mod3.6 consistency definition), so f(n) never decreases
    along a path and no node ever needs to be re-expanded under A*.

    Dominates the zero heuristic used in UCS, so A* with Manhattan expands
    no more nodes than UCS (mod3.6 heuristic dominance).
"""

from typing import Generator, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Grid representation helpers
# ---------------------------------------------------------------------------

def make_grid(rows: int, cols: int,
              obstacles: Optional[set] = None) -> List[List[bool]]:
    """
    Return a rows x cols boolean grid.

    True  = passable cell
    False = obstacle

    Parameters
    ----------
    obstacles : set of (row, col) pairs to block; defaults to empty set
    """
    grid = [[True] * cols for _ in range(rows)]
    for r, c in (obstacles or set()):
        grid[r][c] = False
    return grid


def print_grid(grid: List[List[bool]],
               path: Optional[List[Tuple[int, int]]] = None,
               start: Optional[Tuple[int, int]] = None,
               goal: Optional[Tuple[int, int]] = None) -> None:
    """
    Print a grid to stdout for visual inspection.

    Legend:
      S = start,  G = goal,  * = path,  # = obstacle,  . = open
    """
    path_set = set(path) if path else set()
    rows = len(grid)
    cols = len(grid[0])
    for r in range(rows):
        row_str = ""
        for c in range(cols):
            if (r, c) == start:
                row_str += "S "
            elif (r, c) == goal:
                row_str += "G "
            elif (r, c) in path_set:
                row_str += "* "
            elif not grid[r][c]:
                row_str += "# "
            else:
                row_str += ". "
        print(row_str)
    print()


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------

_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right


class GridProblem:
    """
    Grid pathfinding problem compatible with search.py.

    Exposes the interface expected by best_first_search:
      .initial_state()    -> state
      .goal_test(state)   -> bool
      .successors(state)  -> iterable of (next_state, step_cost)

    The Manhattan distance heuristic is bundled as a method so experiments
    can pass  `problem.manhattan`  directly to astar().
    """

    def __init__(self,
                 grid: List[List[bool]],
                 start: Tuple[int, int],
                 goal: Tuple[int, int]):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.rows = len(grid)
        self.cols = len(grid[0])

    # --- problem interface -------------------------------------------------

    def initial_state(self) -> Tuple[int, int]:
        return self.start

    def goal_test(self, state: Tuple[int, int]) -> bool:
        return state == self.goal

    def successors(self, state: Tuple[int, int]) \
            -> Generator[Tuple[Tuple[int, int], float], None, None]:
        """Yield (next_state, step_cost) for each valid 4-neighbour move."""
        r, c = state
        for dr, dc in _MOVES:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc]:
                yield (nr, nc), 1.0

    # --- heuristics --------------------------------------------------------

    def manhattan(self, state: Tuple[int, int]) -> float:
        """
        Manhattan distance heuristic (mod3.6).

        h(n) = |row_n - row_goal| + |col_n - col_goal|

        Admissible and consistent for 4-neighbour unit-cost grids.
        """
        r, c = state
        gr, gc = self.goal
        return float(abs(r - gr) + abs(c - gc))

    def zero(self, state: Tuple[int, int]) -> float:
        """Zero heuristic -- makes A* equivalent to UCS (used as a baseline)."""
        return 0.0
