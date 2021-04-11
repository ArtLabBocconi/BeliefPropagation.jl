module BeliefPropagation

using Erdos

# imports for FactorGraph definition
using Distributions: Binomial # randgraphs
import Erdos: AGraph, ANetwork, edgetype, vertextype,
              nv, ne, rem_edge!, edge, out_edges,
             out_neighbors, pop_vertex!, add_vertex!,
             add_edge!, swap_vertices!, in_edges

export FactorGraph


export Ising
export Matching
export BMatching

Base.getindex(p::Ptr) = unsafe_load(p)
Base.setindex!(p::Ptr{T}, x::T) where T = unsafe_store!(p, x)

include("factor_graph.jl")
include("matching/Matching.jl")
include("bmatching/BMatching.jl")
include("ising/Ising.jl")

end # module
