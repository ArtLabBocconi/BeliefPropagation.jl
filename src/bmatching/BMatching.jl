
module BMatching
using Random, Statistics

using ExtractMacro
using Printf
using Erdos

export run_bp

getref(v::Vector, i::Integer) = pointer(v, i)

include("bmatching_bp.jl")

end #module
