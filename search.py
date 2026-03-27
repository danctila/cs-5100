"""
Best-first graph search: UCS and A*.

Algorithms follow the pseudocode in:
  mod3.5  -- Uniform-Cost Search (UCS)
  mod3.6  -- A* Search

Both reduce to the same underlying loop; the only difference is the
evaluation function used to order the frontier:

    UCS   : f(n) = g(n)           [mod3.5]
    A*    : f(n) = g(n) + h(n)    [mod3.6]

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
    cost: float         # total path cost  (g value of goal node)
    expansions: int     # nodes removed from frontier and expanded
    peak_frontier: int  # largest frontier size observed during search


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class Node:
    """
    A search node as described in the course lectures (mod3.0 / AIMA ch.3).

    Stores:
      state    -- the world state this node represents
      parent   -- the node that generated this one (None for the root)
      g        -- path cost from the start node to this node
    """

    def __init__(self, state, parent: Optional["Node"] = None, g: float = 0.0):
        self.state = state
        self.parent = parent
        self.g = g

    def reconstruct_path(self) -> List:
        """Follow parent pointers back to the root and return the state sequence."""
        path = []
        node: Optional[Node] = self
        while node is not None:
            path.append(node.state)
            node = node.parent
        return list(reversed(path))

    # heapq needs a tiebreaker; tie on g so earlier-generated nodes go first
    def __lt__(self, other: "Node") -> bool:
        return self.g < other.g


# ---------------------------------------------------------------------------
# General best-first graph search
# ---------------------------------------------------------------------------

def best_first_search(problem, h: Callable = None) -> Optional[SearchResult]:
    """
    General best-first graph search (mod3.6 A* pseudocode generalised).

    Frontier is a min-heap ordered by f(n) = g(n) + h(n).
    Explored set (CLOSED list) prevents re-expanding already-expanded nodes.

    When a cheaper path to a frontier node is found, a new heap entry is
    pushed and the stale one is skipped on pop (lazy deletion), matching
    the 'replace that node with child' step in the course pseudocode.

    Parameters
    ----------
    problem : object with .initial_state(), .goal_test(s), .successors(s)
    h       : heuristic callable h(state) -> float; defaults to zero (UCS)
    """
    if h is None:
        h = lambda s: 0.0

    start_node = Node(state=problem.initial_state(), parent=None, g=0.0)

    # frontier: (f-value, tie-counter, Node)
    # tie-counter ensures FIFO ordering among equal-f nodes (consistent
    # with the course worked examples where ties are broken by insertion order)
    _counter = 0
    frontier: List[Tuple] = []
    heapq.heappush(frontier, (h(start_node.state), _counter, start_node))

    # best known g-value for every state currently in the frontier
    frontier_g = {start_node.state: 0.0}

    explored = set()          # CLOSED list
    expansions = 0
    peak_frontier = 1

    while frontier:
        peak_frontier = max(peak_frontier, len(frontier))

        f_val, _, node = heapq.heappop(frontier)

        # --- lazy-deletion: skip stale entries ---
        # A node is stale if it has already been expanded (in explored)
        # or if a cheaper path to its state was pushed later.
        if node.state in explored:
            continue
        if node.g > frontier_g.get(node.state, float("inf")):
            continue

        # --- goal test (applied at expansion, not generation) ---
        # This matches the course pseudocode and ensures the optimality
        # guarantee holds for UCS and A* (mod3.5 / mod3.6).
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

            # only insert / update if this is a strictly better path
            if g_new < frontier_g.get(next_state, float("inf")):
                frontier_g[next_state] = g_new
                child = Node(state=next_state, parent=node, g=g_new)
                _counter += 1
                heapq.heappush(frontier, (g_new + h(next_state), _counter, child))

    return None  # failure -- no path exists


# ---------------------------------------------------------------------------
# Named wrappers
# ---------------------------------------------------------------------------

def ucs(problem) -> Optional[SearchResult]:
    """
    Uniform-Cost Search (mod3.5).

    Expands nodes in order of increasing path cost g(n).
    Equivalent to A* with h(n) = 0.
    Optimal and complete for nonneg step costs.
    """
    return best_first_search(problem, h=lambda s: 0.0)


def astar(problem, h: Callable) -> Optional[SearchResult]:
    """
    A* Search (mod3.6).

    Expands nodes in order of f(n) = g(n) + h(n).
    Optimal when h is admissible (h(n) <= true remaining cost for all n).
    With a consistent heuristic, no node is ever re-expanded.
    """
    return best_first_search(problem, h=h)
