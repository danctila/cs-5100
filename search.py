"""
Best-first graph search: UCS, A*, and Greedy Best-First.

Algorithms follow the pseudocode in:
  mod3.5  -- Uniform-Cost Search (UCS)
  mod3.6  -- A* Search
  mod3.6  -- Greedy Best-First Search

All three reduce to the same underlying loop. The only difference is
the evaluation function used to order the frontier:

    UCS    : f(n) = g(n)           [mod3.5]
    A*     : f(n) = g(n) + h(n)    [mod3.6]
    Greedy : f(n) = h(n)           [mod3.6]

When h(n) = 0 for all n, A* is identical to UCS.
"""

import heapq
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """Holds everything measured for one search run."""
    path: List          # ordered list of states from start to goal
    cost: float         # total path cost (g value of goal node)
    expansions: int     # nodes removed from the frontier and expanded
    peak_frontier: int  # largest frontier size observed during the search


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class Node:
    """
    A search node as described in the course lectures (mod3.0 / AIMA ch.3).

    Stores:
      state   -- the world state this node represents
      parent  -- the node that generated this one (None for root)
      g       -- path cost from the start node to this node
    """

    def __init__(self, state, parent: Optional["Node"] = None, g: float = 0.0):
        self.state = state
        self.parent = parent
        self.g = g

    def reconstruct_path(self) -> List:
        """Follow parent pointers back to the root; return the state sequence."""
        path = []
        node: Optional[Node] = self
        while node is not None:
            path.append(node.state)
            node = node.parent
        return list(reversed(path))

    # heapq needs __lt__ to break ties; secondary sort on g keeps
    # equal-priority nodes in a deterministic order
    def __lt__(self, other: "Node") -> bool:
        return self.g < other.g


# ---------------------------------------------------------------------------
# General best-first graph search
# ---------------------------------------------------------------------------

def best_first_search(
    problem,
    h: Callable = None,
    ignore_g: bool = False,
) -> Optional[SearchResult]:
    """
    General best-first graph search (mod3.6 A* pseudocode, generalised).

    Frontier  : min-heap ordered by the evaluation function f(n).
    CLOSED    : explored set preventing re-expansion of already-expanded nodes.

    Two modes controlled by `ignore_g`:
      ignore_g=False  ->  f(n) = g(n) + h(n)   (A* / UCS)
      ignore_g=True   ->  f(n) = h(n)           (Greedy Best-First, mod3.6)

    For A*/UCS, when a cheaper path to a frontier node is found, a new heap
    entry is pushed and the stale one is skipped on pop (lazy deletion).
    This matches the 'else if child in frontier with higher g: replace' step
    in the course pseudocode.

    For Greedy, a state is inserted at most once (matching the course
    pseudocode which only inserts 'if not in explored and not in frontier').

    Parameters
    ----------
    problem   : object with .initial_state(), .goal_test(s), .successors(s)
    h         : heuristic callable h(state) -> float; defaults to zero (UCS)
    ignore_g  : if True, priority = h(n) only  (Greedy mode)
    """
    if h is None:
        h = lambda s: 0.0

    start_node = Node(state=problem.initial_state(), parent=None, g=0.0)

    # frontier: (f-value, insertion-counter, Node)
    # insertion counter breaks ties so the heap comparison never reaches Node
    _counter = 0
    frontier: List[Tuple] = []
    heapq.heappush(frontier, (h(start_node.state), _counter, start_node))

    # frontier_g tracks the best known g for each state in the frontier.
    # For Greedy it simply records which states have been inserted (g unused).
    frontier_g = {start_node.state: 0.0}

    explored = set()          # CLOSED list
    expansions = 0
    peak_frontier = 1

    while frontier:
        peak_frontier = max(peak_frontier, len(frontier))

        _, _, node = heapq.heappop(frontier)

        # --- lazy deletion: skip stale entries ---------------------------------
        if node.state in explored:
            continue
        # For A*/UCS only: skip if a cheaper path was inserted later
        if not ignore_g and node.g > frontier_g.get(node.state, float("inf")):
            continue

        # --- goal test applied at expansion, not generation -------------------
        # This is required for the optimality guarantee (mod3.5 / mod3.6):
        # we must be certain the popped node has the lowest f before stopping.
        if problem.goal_test(node.state):
            return SearchResult(
                path=node.reconstruct_path(),
                cost=node.g,
                expansions=expansions,
                peak_frontier=peak_frontier,
            )

        explored.add(node.state)
        expansions += 1

        for next_state, step_cost in problem.successors(node.state):
            if next_state in explored:
                continue

            g_new = node.g + step_cost
            h_val = h(next_state)

            if ignore_g:
                # Greedy: insert each state at most once (mod3.6 pseudocode)
                if next_state not in frontier_g:
                    frontier_g[next_state] = g_new   # track g for path cost
                    child = Node(state=next_state, parent=node, g=g_new)
                    _counter += 1
                    heapq.heappush(frontier, (h_val, _counter, child))
            else:
                # A*/UCS: only push if this is a strictly better path to state
                if g_new < frontier_g.get(next_state, float("inf")):
                    frontier_g[next_state] = g_new
                    child = Node(state=next_state, parent=node, g=g_new)
                    _counter += 1
                    heapq.heappush(frontier, (g_new + h_val, _counter, child))

    return None  # failure -- no path exists


# ---------------------------------------------------------------------------
# Named wrappers (each matches a named algorithm in the course lectures)
# ---------------------------------------------------------------------------

def ucs(problem) -> Optional[SearchResult]:
    """
    Uniform-Cost Search (mod3.5).

    f(n) = g(n). Expands nodes in order of increasing path cost.
    Equivalent to A* with h(n) = 0 for all n.
    Complete and optimal for graphs with nonnegative step costs.
    """
    return best_first_search(problem, h=lambda s: 0.0, ignore_g=False)


def astar(problem, h: Callable) -> Optional[SearchResult]:
    """
    A* Search (mod3.6).

    f(n) = g(n) + h(n). Optimal when h is admissible.
    With a consistent heuristic, no node is ever re-expanded (mod3.6).
    """
    return best_first_search(problem, h=h, ignore_g=False)


def greedy(problem, h: Callable) -> Optional[SearchResult]:
    """
    Greedy Best-First Search (mod3.6).

    f(n) = h(n). Ignores path cost accumulated so far.
    Not optimal and not complete in general (mod3.6 properties).
    Typically expands fewer nodes than A* but may return suboptimal paths.
    """
    return best_first_search(problem, h=h, ignore_g=True)
