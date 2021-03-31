# BeliefPropagation.jl

Implementation of Belief Propagation (BP) message passing for:

- Ising model (`Ising` module)
- Minimum weight perfect matching (`Matching` module)

## Installation

```julia
]add 
https://github.com/CarloLucibello/BeliefPropagation.jl
```

## Usage

Problem instances can be created using [Erdos](
https://github.com/CarloLucibello/Erdos.jl) graph library. Problems are passed to the BP methods as Network objects, with features attached to edges and vertices.

### Ising

```julia
using BeliefPropagation.Ising
using Erdos 

# Create Ising model on ErdosRenyi graph
# with random couplings and constant external field.
net = erdos_renyi(100, 0.02, Network)
eprop!(net, "J", EdgeMap(net, e -> randn()))
vprop!(net, "H", VertexMap(net, v -> 1))

## Run Belief Propagation at some temperature
### and extract magnetizations
fg = Ising.run_bp(net, T=2, maxiters=100);
fg.mags
```

### Matching

```julia
using BeliefPropagation.Matching
using Erdos 

# Create an instance of the random assignment problem
net = CompleteBipartiteGraph(100, 100, Network)
eprop!(net, "w", EdgeMap(net, e -> rand()))

## Run Belief Propagation and obtain optimal cost and matching
E, match = Matching.run_bp(net, maxiters=100);
```

## Related Packages

- [SAT.jl](https://github.com/CarloLucibello/SAT.jl): a BP solver for SAT problems.
- [ForneyLab.jl](https://github.com/biaslab/ForneyLab.jl): Bayesian inference algorithms through message passing on Forney-style factor graphs.
- [BinaryCommitteeMachineFBP.jl](BinaryCommitteeMachineFBP.jl): Focusing Belief Propagation on Commitee machines with binary weights.
