module BeliefPropagation

export Ising
export Matching

Base.getindex(p::Ptr) = unsafe_load(p)
Base.setindex!(p::Ptr{T}, x::T) where T = unsafe_store!(p, x)


module Matching
    using Random, Statistics

    getref(v::Vector, i::Integer) = pointer(v, i)

    include("matching/matching_bp.jl")
end

module Ising
    using Erdos
    using Random, Statistics
    using ExtractMacro
    using Printf
    
    getref(v::Vector, i::Integer) = pointer(v, i)

    include("ising/ising_bp.jl")
    include("ising/ising_tap.jl")
    include("ising/learning_hopfield.jl")
end


end # module
