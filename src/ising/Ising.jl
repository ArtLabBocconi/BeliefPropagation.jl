module Ising

using Erdos
using Random, Statistics
using UnPack
using Printf

export run_bp

getref(v::Vector, i::Integer) = pointer(v, i)

include("ising_bp.jl")
include("ising_tap.jl")

end #module