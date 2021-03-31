
module Matching
using Random, Statistics

using ExtractMacro
using Printf
using Erdos

export run_bp

getref(v::Vector, i::Integer) = pointer(v, i)

include("matching_bp.jl")

end #module
