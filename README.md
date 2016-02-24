# BeliefPropagation
Poorly tested, use it at your own risk.

## KSAT
Solve random instance with BP inspired procedures.
`solveKSAT` is the main function, and a solver `method` can be chosen
beetween `:reinforcement` (default),  `:decimation`

### Reinforcement
`r` is the initial value of the reinforcement parameter (`r=0.` default).
`r_step` determines its moltiplicative increment.
```julia
σ = KSAT.solve(N=10000, α=9.6, k=4, seed_cnf=19, r_step=0.0002, maxiters=1000);
```

Read file in CNF format and solve with BP + reinforcement
```julia
σ = KSAT.solve("file.cnf", r_step=0.01, maxiters=1000);
```

If having errors, try to reduce `reinf_step`.

### Decimation
After each convergence of the BP algorithm the `r*N` most biased variables are fixed.
```julia
σ = KSAT.solve(method=:decimation, N=10000,α=9.6, k=4, seed_cnf=19, r=0.02, maxiters=1000);
```

## Ising
BP on pairwise Ising. Preliminary work

## Perceptron
### BP + Reinforcement
BP + reinforcement to solve binary perceptron.
```julia
W = Perceptron.solve(N=1001,α=0.6, seed_ξ=19, r=0.6, r_step=0.001, maxiters=1000);
```
### TAP + Reinforcement
TAP + reinforcement to solve binary perceptron.
```julia
W = PerceptronTAP.solve(N=1001,α=0.7, r=0.4, r_step=0.001, maxiters=1000);
```
### EdTAP + Reinforcement
Work In Progress : Entropy driven TAP for binary perceptron.
```julia
W = PerceptronTAP.solve(N=1001,α=0.7, γ=0.4, y=4., maxiters=1000);
