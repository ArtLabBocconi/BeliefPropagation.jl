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