"""
Sanity tests for Week 2 of the capstone.

Each test verifies a claim made in the Milestone 2 report:

  1. Both UCS and A* return the known optimal path cost.
  2. A* (Manhattan) expands <= nodes than UCS on the same instance.
  3. Both algorithms correctly handle obstacles (must route around them).
  4. No path exists returns None (failure).
  5. A* with a stronger heuristic expands <= nodes than A* with a weaker one
     -- validates the dominance claim from mod3.6.

Run with:
    python run_tests.py
"""

import sys
from grid import GridProblem, make_grid, print_grid
from search import ucs, astar

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def check(label: str, condition: bool, detail: str = "") -> bool:
    tag = PASS if condition else FAIL
    msg = f"[{tag}] {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return condition


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def run_both(problem):
    """Run UCS and A*(Manhattan) on the same problem and return both results."""
    r_ucs = ucs(problem)
    r_astar = astar(problem, problem.manhattan)
    return r_ucs, r_astar


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_trivial_same_cell():
    """Start == goal: cost 0, path length 1, 0 expansions."""
    grid = make_grid(3, 3)
    prob = GridProblem(grid, (1, 1), (1, 1))
    r = ucs(prob)
    check("trivial: start==goal, cost==0",
          r is not None and r.cost == 0 and len(r.path) == 1,
          f"cost={r.cost if r else 'None'}")


def test_3x3_open():
    """
    3x3 open grid, (0,0) -> (2,2).

    Manhattan distance = 4, which is also the optimal path cost
    (4 steps in a 4-neighbour grid).
    """
    grid = make_grid(3, 3)
    prob = GridProblem(grid, (0, 0), (2, 2))
    r_ucs, r_astar = run_both(prob)

    all_ok = True
    all_ok &= check("3x3 open: UCS optimal cost == 4",
                    r_ucs is not None and r_ucs.cost == 4,
                    f"cost={r_ucs.cost if r_ucs else 'None'}")
    all_ok &= check("3x3 open: A* optimal cost == 4",
                    r_astar is not None and r_astar.cost == 4,
                    f"cost={r_astar.cost if r_astar else 'None'}")
    all_ok &= check("3x3 open: A* expansions <= UCS expansions",
                    r_astar.expansions <= r_ucs.expansions,
                    f"A*={r_astar.expansions}, UCS={r_ucs.expansions}")

    if all_ok:
        print("  Path (A*):", r_astar.path)
    print()


def test_5x5_obstacle():
    """
    5x5 grid with a vertical wall forcing a detour.

      . . # . .
      . . # . .
      . . # . .
      . . . . .
      . . . . .

    Start=(0,0), Goal=(0,4). Straight path is blocked; must go around.
    Optimal cost = 8 (down 3, right 3, up 3 -- or symmetric variants).

    Actually let me compute: from (0,0) to (0,4) with col-2 blocked for rows 0-2.
    One optimal path: (0,0)->(1,0)->(2,0)->(3,0)->(3,1)->(3,2)->(3,3)->(3,4)->(2,4)->(1,4)->(0,4) = 10 steps
    
    Hmm, let me recalculate. Blocking col=2 rows 0-2 means:
    row 3 has col 2 open, so we go around via row 3.
    Path: right to col 1, down to row 3, right to col 4, up to row 0 = 1+3+3+3 = 10? No.
    (0,0) -> right -> (0,1) -> down -> (1,1) -> down -> (2,1) -> down -> (3,1) -> right -> (3,2) -> right -> (3,3) -> right -> (3,4) -> up -> (2,4) -> up -> (1,4) -> up -> (0,4)
    = 17 steps? That seems too many.
    
    Let me reconsider. Simpler: just go down 3, then across, then up, but only in the cols that are open.
    Actually from (0,0) to (0,4):
    Direct path would be 4 steps going right, but col 2 is blocked for rows 0-2.
    
    Going around: (0,0)->(1,0)->(2,0)->(3,0)->(3,1)->(3,2)->(3,3)->(3,4)->(2,4)->(1,4)->(0,4) = 10 steps.
    
    But can we go through rows 0-2 with cols != 2? Like:
    (0,0)->(0,1) then we need col 2 which is blocked...
    Or (0,0)->(1,0)->(1,1) then (1,2) is blocked...
    
    So minimum path goes through row 3 where col 2 is open: cost = 10.
    
    Actually let me use a simpler obstacle setup for the test to make it easy to verify.
    Let me use a straight wall: col 2 blocked for ALL rows, except... or a simpler scenario.
    
    Let me use a 5x5 grid with just one obstacle at (0,2):
    Start=(0,0), Goal=(0,4). Path must detour around (0,2).
    One path: (0,0)->(0,1)->(1,1)->(1,2)->(1,3)->(0,3)->(0,4) = 6 steps.
    Manhattan direct = 4, but one obstacle forces a detour.
    
    With obstacle at (0,2), detour of 2 extra steps = cost 6.
    Path: right, down, right, right, up, right = 6.
    
    This is easy to verify!
    """
    # Block (0,2) only -- forces detour on the direct row
    grid = make_grid(5, 5, obstacles={(0, 2)})
    prob = GridProblem(grid, (0, 0), (0, 4))
    r_ucs, r_astar = run_both(prob)

    # optimal: (0,0)->(0,1)->(1,1)->(1,2)->(1,3)->(0,3)->(0,4) = cost 6
    all_ok = True
    all_ok &= check("5x5 obstacle: UCS optimal cost == 6",
                    r_ucs is not None and r_ucs.cost == 6,
                    f"cost={r_ucs.cost if r_ucs else 'None'}")
    all_ok &= check("5x5 obstacle: A* optimal cost == 6",
                    r_astar is not None and r_astar.cost == 6,
                    f"cost={r_astar.cost if r_astar else 'None'}")
    all_ok &= check("5x5 obstacle: A* expansions <= UCS expansions",
                    r_astar.expansions <= r_ucs.expansions,
                    f"A*={r_astar.expansions}, UCS={r_ucs.expansions}")

    if all_ok:
        print("  Path (A*):", r_astar.path)
        print_grid(grid, path=r_astar.path, start=(0, 0), goal=(0, 4))
    print()


def test_no_path():
    """Goal is completely surrounded by obstacles -- expect failure (None)."""
    grid = make_grid(5, 5, obstacles={(1, 2), (2, 1), (2, 3), (3, 2)})
    prob = GridProblem(grid, (0, 0), (2, 2))
    r = ucs(prob)
    check("no path: UCS returns None",
          r is None,
          f"got cost={r.cost if r else 'None'}")
    print()


def test_10x10_open():
    """
    10x10 open grid, (0,0) -> (9,9).

    Optimal cost = 18 (Manhattan distance).
    A* must expand strictly fewer nodes than UCS.
    """
    grid = make_grid(10, 10)
    prob = GridProblem(grid, (0, 0), (9, 9))
    r_ucs, r_astar = run_both(prob)

    all_ok = True
    all_ok &= check("10x10 open: UCS optimal cost == 18",
                    r_ucs is not None and r_ucs.cost == 18,
                    f"cost={r_ucs.cost if r_ucs else 'None'}")
    all_ok &= check("10x10 open: A* optimal cost == 18",
                    r_astar is not None and r_astar.cost == 18,
                    f"cost={r_astar.cost if r_astar else 'None'}")
    all_ok &= check("10x10 open: A* expansions <= UCS expansions",
                    r_astar.expansions <= r_ucs.expansions,
                    f"A*={r_astar.expansions}, UCS={r_ucs.expansions}")
    all_ok &= check("10x10 open: A* peak frontier <= UCS peak frontier",
                    r_astar.peak_frontier <= r_ucs.peak_frontier,
                    f"A*={r_astar.peak_frontier}, UCS={r_ucs.peak_frontier}")
    print()


def test_heuristic_dominance():
    """
    Dominance test (mod3.6): stronger h expands <= nodes than weaker h.

    We compare:
      A* with zero heuristic   (equivalent to UCS)
      A* with Manhattan        (stronger, admissible)

    The dominance property from the course guarantees:
      expansions(A* with Manhattan) <= expansions(A* with zero)
    """
    grid = make_grid(20, 20)
    prob = GridProblem(grid, (0, 0), (19, 19))

    r_weak = astar(prob, prob.zero)       # h=0, same as UCS
    r_strong = astar(prob, prob.manhattan)  # h=Manhattan

    all_ok = True
    all_ok &= check("dominance: both return same (optimal) cost",
                    r_weak is not None and r_strong is not None
                    and r_weak.cost == r_strong.cost,
                    f"weak={r_weak.cost if r_weak else 'None'}, "
                    f"strong={r_strong.cost if r_strong else 'None'}")
    all_ok &= check("dominance: strong h expands <= weak h",
                    r_strong.expansions <= r_weak.expansions,
                    f"strong={r_strong.expansions}, weak={r_weak.expansions}")
    print()


def test_path_is_valid(n_random: int = 10, seed: int = 42):
    """
    For n_random random 15x15 grids, verify:
      - A* and UCS return the same cost (both optimal).
      - Returned path is actually connected (every step is a valid move).
    """
    import random
    rng = random.Random(seed)
    passed = 0

    for i in range(n_random):
        rows, cols = 15, 15
        n_obstacles = rng.randint(10, 40)
        obstacles = set()
        while len(obstacles) < n_obstacles:
            r = rng.randint(0, rows - 1)
            c = rng.randint(0, cols - 1)
            if (r, c) not in {(0, 0), (rows - 1, cols - 1)}:
                obstacles.add((r, c))

        grid = make_grid(rows, cols, obstacles)
        prob = GridProblem(grid, (0, 0), (rows - 1, cols - 1))

        r_ucs = ucs(prob)
        r_astar = astar(prob, prob.manhattan)

        # both should agree on cost (or both fail)
        if r_ucs is None and r_astar is None:
            passed += 1
            continue

        if r_ucs is None or r_astar is None:
            print(f"  [FAIL] instance {i}: one found a path, the other did not")
            continue

        costs_match = abs(r_ucs.cost - r_astar.cost) < 1e-9
        astar_le_ucs = r_astar.expansions <= r_ucs.expansions

        # verify A* path is actually valid (connected)
        path = r_astar.path
        valid_path = True
        for j in range(len(path) - 1):
            r1, c1 = path[j]
            r2, c2 = path[j + 1]
            if abs(r1 - r2) + abs(c1 - c2) != 1:
                valid_path = False
                break
            if not grid[r2][c2]:
                valid_path = False
                break

        if costs_match and astar_le_ucs and valid_path:
            passed += 1

    check(f"random grids ({n_random} instances): all pass cost/dominance/path-validity",
          passed == n_random,
          f"{passed}/{n_random} passed")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Capstone Week 2 -- Search sanity tests")
    print("=" * 60)
    print()

    test_trivial_same_cell()
    test_3x3_open()
    test_5x5_obstacle()
    test_no_path()
    test_10x10_open()
    test_heuristic_dominance()
    test_path_is_valid()

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
