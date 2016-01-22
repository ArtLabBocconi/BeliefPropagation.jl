module BeliefPropagation
using LightGraphs
import Base.show
export KSAT, Ising
export Perceptron, PerceptronTAP, PerceptronETAP

Base.getindex(p::Ptr) = unsafe_load(p)
Base.setindex!{T}(p::Ptr{T}, x::T) = unsafe_store!(p, x)
Base.show(io::IO, p::Ptr) = show(io, p[])
Base.show(p::Ptr) = show(p[])

module KSAT
    include("ksat.jl")
end

module Ising
    include("ising.jl")
    include("tap_ising.jl")
    include("mc_ising.jl")
    include("learning_hopfield.jl")
end

module Perceptron
    include("perceptron.jl")
end

module PerceptronTAP
    include("tap_perceptron.jl")
end

module PerceptronEdTAP
    include("edtap_perceptron.jl")
end


end # module
