Key observations:

1. Optimality: UCS, A*(h1), A*(h2) return optimal costs on every instance.
   Greedy returns suboptimal paths on some instances (as expected from
   mod3.6: Greedy is not optimal in general).

2. Heuristic dominance (mod3.6): The expansion ordering holds across all
   instances: UCS >= A*(h1) >= A*(h2).
   This reproduces the paper's claim that a stronger admissible heuristic
   h2 >= h1 implies A*(h2) expands no more nodes than A*(h1).

3. Greedy trade-off: Greedy typically expands the fewest nodes but
   produces suboptimal paths. This shows why A\* includes g(n)
   because it prevents committing to cheap-looking paths that are actually long.
