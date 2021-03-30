using BeliefPropagation
using Test
using Random, Statistics
using Erdos
using RRRMC

include("test_utils.jl")

@testset "Matching" begin
    include("matching.jl")
end
@testset "Ising" begin
    include("ising.jl")
end
