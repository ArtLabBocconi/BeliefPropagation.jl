using BeliefPropagation
using Test

@testset "BeliefPropagation.jl" begin 
    @testset "Matching" begin
        include("matching.jl")
    end
    @testset "Ising" begin
        include("ising.jl")
    end
end
