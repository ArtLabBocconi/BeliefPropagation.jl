using BeliefPropagation
using Test
using Random, Statistics

include("test_utils.jl")

@testset "Matching" begin
    include("matching.jl")
end
@testset "Ising" begin
    include("ising.jl")
end
