using BeliefPropagation
using Test
using Random, Statistics
using Erdos
using RRRMC
import OnlineStats

include("test_utils.jl")

@testset "Matching" begin
    include("matching.jl")
end

@testset "B-Matching" begin
    include("bmatching.jl")
end

@testset "Ising" begin
    include("ising.jl")
end
