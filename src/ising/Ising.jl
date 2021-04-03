module Ising

using Erdos
using Random, Statistics
using ExtractMacro
using Printf

export run_bp

abstract type FactorGraph end

getref(v::Vector, i::Integer) = pointer(v, i)

include("ising_bp.jl")
include("ising_tap.jl")

end #module