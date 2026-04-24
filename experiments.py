"""
Capstone Experiments Weeks 3 & 4.

Runs all four algorithms on both domains and prints summary tables.
Results are also saved to results.txt in this directory.

Domains
-------
Grid pathfinding (mod3.0 / mod3.5 / mod3.6):
    4-neighbour, unit-cost grid with random obstacles.
    Algorithms: UCS, Greedy(Manhattan), A*(Manhattan)

8-puzzle (mod3.6 "Heuristics for the 8-Puzzle"):
    Classic 15-state sliding-tile problem.
    Algorithms: UCS, A*(h1=misplaced), A*(h2=Manhattan), Greedy(h2)
    Depths tested: 8 and 14 moves from goal.
    NOTE: UCS is skipped at depth 14 -- at that depth A*(h1) already
    needs ~539 expansions on average (mod3.6 table), so UCS would
    require tens of thousands and is impractical for a demo.

Metrics
-------
  path_cost - g value of goal node (total moves / steps)
  expansions - nodes removed from frontier and expanded
  optimal - True if cost matches the known-optimal cost (from A*(h2))

Success criteria (Milestone 2, Section 4)
-----------------------------------------
  UCS, A*(h1), A*(h2) return optimal costs on every instance.
  Node expansion ordering: UCS >= A*(h1) >= A*(h2)  (heuristic dominance)
  Greedy expands fewer nodes than A* on some instances but is
  sometimes suboptimal.
"""

import random
import sys
from statistics import mean, stdev
from typing import Callable, Dict, List, Optional

from grid import GridProblem, make_grid
from puzzle import PuzzleProblem, GOAL, generate_batch
from search import ucs, astar, greedy, SearchResult


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def fmt(val: Optional[float], decimals: int = 1) -> str:
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def format_table(rows: List[Dict], col_order: List[str], col_width: int = 16) -> List[str]:
    header = "".join(c.ljust(col_width) for c in col_order)
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append("".join(str(row.get(c, "")).ljust(col_width) for c in col_order))
    return lines


# ---------------------------------------------------------------------------
# Grid experiments
# ---------------------------------------------------------------------------

def run_grid_experiments(
    size: int,
    n_instances: int,
    seed: int,
    obstacle_frac: float = 0.15,
) -> Dict:
    """
    Run UCS, Greedy, and A*(Manhattan) on `n_instances` random grids.

    Grid generation: each cell is blocked independently with probability
    obstacle_frac; start=(0,0) and goal=(size-1,size-1) are always open.
    Instances where no path exists are skipped (counted separately).
    """
    rng = random.Random(seed)
    results = {alg: {"costs": [], "expansions": []} for alg in
               ["UCS", "Greedy", "A*(Manhattan)"]}
    skipped = 0

    for _ in range(n_instances):
        obstacles = set()
        for r in range(size):
            for c in range(size):
                if (r, c) in {(0, 0), (size - 1, size - 1)}:
                    continue
                if rng.random() < obstacle_frac:
                    obstacles.add((r, c))

        grid = make_grid(size, size, obstacles)
        prob = GridProblem(grid, (0, 0), (size - 1, size - 1))

        r_ucs = ucs(prob)
        if r_ucs is None:
            skipped += 1
            # this should skip unsolvable instances
            continue   

        r_greedy = greedy(prob, prob.manhattan)
        r_astar  = astar(prob, prob.manhattan)

        results["UCS"]["costs"].append(r_ucs.cost)
        results["UCS"]["expansions"].append(r_ucs.expansions)

        # Greedy may fail on some obstacle configs (bc not complete in general)
        if r_greedy is not None:
            results["Greedy"]["costs"].append(r_greedy.cost)
            results["Greedy"]["expansions"].append(r_greedy.expansions)
        else:
            results["Greedy"]["costs"].append(None)
            results["Greedy"]["expansions"].append(None)

        results["A*(Manhattan)"]["costs"].append(r_astar.cost)
        results["A*(Manhattan)"]["expansions"].append(r_astar.expansions)

    return results, skipped


def summarize_grid(results: Dict, n_instances: int, skipped: int, size: int) -> List[Dict]:
    """Build table rows; check dominance and optimality properties."""
    rows = []
    # optimal baseline
    ref_costs = results["A*(Manhattan)"]["costs"]   

    for alg_name, data in results.items():
        costs = [c for c in data["costs"] if c is not None]
        exps  = [e for e in data["expansions"] if e is not None]
        ref   = [r for r, c in zip(ref_costs, data["costs"]) if c is not None]

        n_subopt = sum(1 for c, r in zip(costs, ref) if abs(c - r) > 1e-9)
        n_fail   = sum(1 for c in data["costs"] if c is None)

        rows.append({
            "Algorithm":     alg_name,
            "Avg Cost":      fmt(mean(costs)) if costs else "N/A",
            "Avg Expansions": fmt(mean(exps), 0) if exps else "N/A",
            "Suboptimal":    f"{n_subopt}/{len(costs)}",
            "Failed":        str(n_fail),
        })
    return rows


# ---------------------------------------------------------------------------
# 8-puzzle experiments
# ---------------------------------------------------------------------------

def run_puzzle_experiments(
    depth: int,
    n_instances: int,
    seed: int,
    skip_ucs: bool = False,
) -> Dict:
    """
    Run algorithms on `n_instances` 8-puzzle instances generated at `depth`.

    skip_ucs: set True for depth=14 where UCS is intractably slow.
    The mod3.6 table shows A*(h1) needs ~539 expansions at depth 14;
    UCS would need orders of magnitude more (exponential in depth).
    """
    instances = generate_batch(n_instances, depth, seed)
    results = {alg: {"costs": [], "expansions": []}
               for alg in ["UCS", "A*(h1)", "A*(h2)", "Greedy(h2)"]}

    for start_state in instances:
        prob = PuzzleProblem(start_state)

        if not skip_ucs:
            r_ucs = ucs(prob)
            if r_ucs is not None:
                results["UCS"]["costs"].append(r_ucs.cost)
                results["UCS"]["expansions"].append(r_ucs.expansions)
            else:
                results["UCS"]["costs"].append(None)
                results["UCS"]["expansions"].append(None)

        r_h1     = astar(prob, PuzzleProblem.h1)
        r_h2     = astar(prob, PuzzleProblem.h2)
        r_greedy = greedy(prob, PuzzleProblem.h2)

        results["A*(h1)"]["costs"].append(r_h1.cost if r_h1 else None)
        results["A*(h1)"]["expansions"].append(r_h1.expansions if r_h1 else None)

        results["A*(h2)"]["costs"].append(r_h2.cost if r_h2 else None)
        results["A*(h2)"]["expansions"].append(r_h2.expansions if r_h2 else None)

        results["Greedy(h2)"]["costs"].append(r_greedy.cost if r_greedy else None)
        results["Greedy(h2)"]["expansions"].append(r_greedy.expansions if r_greedy else None)

    if skip_ucs:
        del results["UCS"]

    return results


def summarize_puzzle(results: Dict, depth: int) -> List[Dict]:
    """Build table rows; compare to mod3.6 published figures."""
    # mod3.6 table (nodes generated, not expanded; our metric differs slightly)
    # We cite these as reference for the trend, not for exact matching.
    course_ref = {
        8:  {"A*(h1)": 39,  "A*(h2)": 25},
        14: {"A*(h1)": 539, "A*(h2)": 113},
    }

    ref_costs = results.get("A*(h2)", {}).get("costs", [])
    rows = []

    for alg_name, data in results.items():
        costs = [c for c in data["costs"] if c is not None]
        exps  = [e for e in data["expansions"] if e is not None]
        ref   = [r for r, c in zip(ref_costs, data["costs"]) if c is not None]

        n_subopt = sum(1 for c, r in zip(costs, ref) if abs(c - r) > 1e-9)

        course_val = course_ref.get(depth, {}).get(alg_name, "—")

        rows.append({
            "Algorithm":       alg_name,
            "Avg Cost":        fmt(mean(costs), 1) if costs else "N/A",
            "Avg Expansions":  fmt(mean(exps), 1) if exps else "N/A",
            "mod3.6 Ref":      str(course_val),
            "Suboptimal":      f"{n_subopt}/{len(costs)}",
        })
    return rows


# ---------------------------------------------------------------------------
# Dominance checker
# ---------------------------------------------------------------------------

def check_dominance(results: Dict, alg_weak: str, alg_strong: str) -> str:
    """
    Verify that alg_strong expands <= nodes than alg_weak on every instance.
    Returns a pass/fail summary string.

    This reproduces the heuristic dominance claim from mod3.6:
    'A* with h2 expands no more nodes than A* with h1'
    """
    weak_exp   = results[alg_weak]["expansions"]
    strong_exp = results[alg_strong]["expansions"]
    violations = sum(
        1 for w, s in zip(weak_exp, strong_exp)
        if w is not None and s is not None and s > w
    )
    total = sum(1 for w, s in zip(weak_exp, strong_exp)
                if w is not None and s is not None)
    status = "PASS" if violations == 0 else f"FAIL ({violations} violations)"
    return f"{alg_strong} <= {alg_weak}: {status} ({total} instances)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_lines = []

    def log(line: str = ""):
        print(line)
        output_lines.append(line)

    # Constants
    N_GRID      = 20
    N_PUZZLE    = 20
    SEED        = 42

    log("Capstone Experiments")

    # ------------------------------------------------------------------
    # Grid: 15x15
    # ------------------------------------------------------------------
    log("\n--- Grid Pathfinding 15x15 (20 instances, seed=42) ---")
    log("Algorithms: UCS  |  Greedy(Manhattan)  |  A*(Manhattan)")
    log("All unit-cost 4-neighbour grids; start=(0,0) goal=(14,14)\n")

    grid_15_results, skipped_15 = run_grid_experiments(15, N_GRID, SEED)
    rows_15 = summarize_grid(grid_15_results, N_GRID, skipped_15, 15)
    for line in format_table(rows_15, ["Algorithm", "Avg Cost", "Avg Expansions", "Suboptimal", "Failed"]):
        log(line)
    if skipped_15:
        log(f"\n  (Skipped {skipped_15} unsolvable instances)")

    log(f"\n  Dominance:")
    log("  " + check_dominance(grid_15_results, "UCS", "A*(Manhattan)"))

    # ------------------------------------------------------------------
    # Grid: 30x30
    # ------------------------------------------------------------------
    log("\n--- Grid Pathfinding 30x30 (20 instances, seed=42) ---")
    log("Algorithms: UCS  |  Greedy(Manhattan)  |  A*(Manhattan)\n")

    grid_30_results, skipped_30 = run_grid_experiments(30, N_GRID, SEED)
    rows_30 = summarize_grid(grid_30_results, N_GRID, skipped_30, 30)
    for line in format_table(rows_30, ["Algorithm", "Avg Cost", "Avg Expansions", "Suboptimal", "Failed"]):
        log(line)
    if skipped_30:
        log(f"\n  (Skipped {skipped_30} unsolvable instances)")

    log(f"\n  Dominance:")
    log("  " + check_dominance(grid_30_results, "UCS", "A*(Manhattan)"))

    # ------------------------------------------------------------------
    # 8-puzzle: depth 8
    # ------------------------------------------------------------------
    log("\n--- 8-Puzzle depth~8 (20 instances, seed=42) ---")
    log("Algorithms: UCS  |  A*(h1=misplaced)  |  A*(h2=Manhattan)  |  Greedy(h2)")
    log("mod3.6 reference (nodes generated): A*(h1)=39, A*(h2)=25\n")

    puz_8_results = run_puzzle_experiments(8, N_PUZZLE, SEED, skip_ucs=False)
    rows_puz8 = summarize_puzzle(puz_8_results, depth=8)
    for line in format_table(rows_puz8, ["Algorithm", "Avg Cost", "Avg Expansions", "mod3.6 Ref", "Suboptimal"]):
        log(line)

    log(f"\n  Dominance:")
    log("  " + check_dominance(puz_8_results, "UCS",    "A*(h1)"))
    log("  " + check_dominance(puz_8_results, "A*(h1)", "A*(h2)"))

    # ------------------------------------------------------------------
    # 8-puzzle: depth 14  (UCS skipped, intractable at this depth)
    # ------------------------------------------------------------------
    log("\n--- 8-Puzzle depth~14 (20 instances, seed=42) ---")
    log("Algorithms: A*(h1=misplaced)  |  A*(h2=Manhattan)  |  Greedy(h2)")
    log("UCS omitted: at depth 14, UCS would need ~10,000+ expansions per instance.")
    log("mod3.6 reference (nodes generated): A*(h1)=539, A*(h2)=113\n")

    puz_14_results = run_puzzle_experiments(14, N_PUZZLE, SEED, skip_ucs=True)
    rows_puz14 = summarize_puzzle(puz_14_results, depth=14)
    for line in format_table(rows_puz14, ["Algorithm", "Avg Cost", "Avg Expansions", "mod3.6 Ref", "Suboptimal"]):
        log(line)

    log(f"\n  Dominance:")
    log("  " + check_dominance(puz_14_results, "A*(h1)", "A*(h2)"))

    with open("results.txt", "w") as f:
        f.write("\n".join(output_lines))
    print("\nResults saved to results.txt")


if __name__ == "__main__":
    main()
