module BeliefPropagation
using LightGraphs
import Base.show
export KSATBP, Ising
export PerceptronBP, PerceptronTAP, PerceptronETAP
export CommitteeBP, CommitteeTAP

Base.getindex(p::Ptr) = unsafe_load(p)
Base.setindex!{T}(p::Ptr{T}, x::T) = unsafe_store!(p, x)
Base.show(io::IO, p::Ptr) = show(io, p[])
Base.show(p::Ptr) = show(p[])

module KSATBP
    include("ksat/ksat_bp.jl")
end

module Ising
    include("ising/ising_bp.jl")
    include("ising/ising_tap.jl")
    include("ising/ising_mc.jl")
    include("ising/learning_hopfield.jl")
end

module PerceptronEdTAP
    include("perceptron/perceptron_edtap.jl")
end

include("deeplearning/deep_binary.jl")


end # module
