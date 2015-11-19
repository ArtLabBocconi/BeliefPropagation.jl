# BeliefPropagation
Poorly tested, use it at your own risk.

## KSAT
Solve random instance with BP + reinforcement.
`r` is the value of the reinforcement parameter.
`r_step` determines its moltiplicative increment.
```julia
σ = BeliefPropagation.solveKSAT(N=10000,α=9.6, k=4, seed_cnf=19, r_step=0.0002, maxiters=1000);
```


Read file in CNF format and solve with BP + reinforcement
```julia
σ = BeliefPropagation.solveKSAT("file.cnf", r_step=0.01, maxiters=1000);
```

If having errors, try to reduce `reinf_step`.

## Ising

BP on pairwise Ising. Preliminary work
