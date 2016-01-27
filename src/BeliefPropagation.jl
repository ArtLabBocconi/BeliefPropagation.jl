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
    include("ksat_bp.jl")
end

module Ising
    include("ising_bp.jl")
    include("ising_tap.jl")
    include("ising_mc.jl")
    include("learning_hopfield.jl")
end

module PerceptronBP
    include("perceptron_bp.jl")
end

module PerceptronTAP
    include("perceptron_tap.jl")
end

module PerceptronEdTAP
    include("perceptron_edtap.jl")
end

module CommitteeBP
    #Work In Progress
    include("committee_bp.jl")
end

module CommitteeTAP
    include("committee_tap.jl")
end



end # module
