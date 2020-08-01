using BeliefPropagation
using Test

@testset "BeliefPropagation.jl" begin 
    @testset "Functions" begin
        include("functions.jl")
    end

    @testset "DeepMP" begin
        include("deeplearning.jl")
    end

    @testset "KSAT" begin
        include("ksat.jl")
    end

    @testset "Matching" begin
        include("matching.jl")
    end
end

# @time g, W, E, stab = DeepBinary.solve(α=0.15, K=[301,21,11,3,1]
#                    , layers=[:tap,:tap,:tapex,:bpex]
#                    ,r=.9,rstep=0.0005,ry=0.01, seedξ=1,maxiters=2000);
