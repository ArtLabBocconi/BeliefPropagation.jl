module BeliefPropagation
using LightGraphs
import Base.show
export KSATBP, Ising
export PerceptronEdTAP
export DeepBinary

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


include("deeplearning/deep_binary.jl")


############## EXPERIMENTAL #########################
# module PerceptronEdTAP
#     include("../src/experimental/perceptron/perceptron_edtap.jl")
# end

## ## EdTAP + Reinforcement
## *Work In Progress*
## Entropy driven TAP for binary perceptron.
## ```julia
## W = PerceptronTAP.solve(N=1001,α=0.7, γ=0.4, y=4., maxiters=1000);
## ```

end # module
