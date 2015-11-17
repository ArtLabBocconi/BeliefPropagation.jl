# BeliefPropagation
Poorly tested, use it at your own risk.

## KSAT
Solve random instance with BP + reinforcement
```julia
σ = BeliefPropagation.solveKSAT(N=4000,α=9.2, seed_cnf=19, reinf_step=0.001, maxiters=1000);
```

Read file in CNF format and solve with BP + reinforcement
```julia
σ = BeliefPropagation.solveKSAT("file.cnf", reinf_step=0.01, maxiters=1000);
```
