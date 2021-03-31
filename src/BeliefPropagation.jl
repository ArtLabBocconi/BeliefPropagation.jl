module BeliefPropagation

export Ising
export Matching

Base.getindex(p::Ptr) = unsafe_load(p)
Base.setindex!(p::Ptr{T}, x::T) where T = unsafe_store!(p, x)


include("matching/Matching.jl")
include("ising/Ising.jl")

end # module
