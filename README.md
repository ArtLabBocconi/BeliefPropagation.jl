# BeliefPropagation.jl

![CI](https://github.com/CarloLucibello/BeliefPropagation.jl/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/CarloLucibello/BeliefPropagation.jl/branch/master/graph/badge.svg?token=EWNYPD7ASX)](https://codecov.io/gh/CarloLucibello/BeliefPropagation.jl)

Implementation of Belief Propagation (BP) message passing for:

- Ising model (`Ising` module)
- Minimum weight perfect matching (`Matching` module)
- Minimum weight perfect b-matching (`BMatching` module)

Package is still experimental and not thoroughly tested, use it at your own risk.
Code contributions are very welcome!

## Installation

```julia
]add https://github.com/ArtLabBocconi/BeliefPropagation.jl
```

## Usage

Problem instances can be created using [Erdos](https://github.com/CarloLucibello/Erdos.jl) graph library. Problems are passed to  BP methods as Network objects with features attached to edges and vertices.

### Ising

```julia
using BeliefPropagation.Ising
using Erdos 

# Create Ising model on ErdosRenyi graph
# with random couplings and constant external field.
net = erdos_renyi(100, 0.02, Network)
eprop!(net, "J", e -> randn())
vprop!(net, "H", v -> 1)

## Run Belief Propagation at some temperature
### and extract magnetizations
fg = Ising.run_bp(net, T=2, maxiters=100);
fg.mags
```

### Matching

```julia
using BeliefPropagation.Matching
using Erdos 
using LinearAlgebra

# Create an instance of the random assignment problem
net = CompleteBipartiteGraph(100, 100, Network)
pos = [rand(2) for i in 1:nv(net)]
eprop!(net, "w", e -> norm(pos[src(e)] .- pos[dst(e)]))

## Run Belief Propagation and obtain optimal cost and matching
E, match = Matching.run_bp(net, maxiters=100);
```

The algorithm is guaranteed to return exact solutions only on bipartite graphs
(altough it may also work on non-bipartite).

### B-Matching

Solve an instance of the minimum-weight perfect b-matching problem.

```julia
using BeliefPropagation.BMatching
using Erdos 
using LinearAlgebra

# Create an instance of the random assignment problem
net = CompleteGraph(100, Network)
eprop!(net, "w", e -> rand())
vprop!(net, "b", v -> rand(1:5))

## Run Belief Propagation and obtain optimal cost and matching
E, match = BMatching.run_bp(net, maxiters=100);
```

## Related Packages

- [SAT.jl](https://github.com/CarloLucibello/SAT.jl): a BP solver for SAT problems.
- [ForneyLab.jl](https://github.com/biaslab/ForneyLab.jl): Bayesian inference algorithms through message passing on Forney-style factor graphs.
- [BinaryCommitteeMachineFBP.jl](BinaryCommitteeMachineFBP.jl): Focusing Belief Propagation on commitee machines (neural network with one-hidden layer and a single output) with binary weights.

## TODO

Problems to be implemented:

- coloring
- xorsat
- Potts models
- Steiner tree

Computation of Bethe free energies is also missing.
