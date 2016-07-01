include("../src/BeliefPropagation.jl")
using BeliefPropagation
using Base.Test

println("@ START TESTING")
println("@ Testing Functions...")
# include("functions.jl")
println("@ ... done.")

println("@ Testing DeepLearning...")
# include("deeplearning.jl")
println("@ ... done.")

println("@ Testing KSAT...")
# include("ksat.jl")
println("@ ... done.")

println("@ Testing Matching...")
include("matching.jl")
println("@ ... done.")

println("@ ALL TESTS PASSED!")

# @time g, W, E, stab = DeepBinary.solve(α=0.15, K=[301,21,11,3,1]
#                    , layers=[:tap,:tap,:tapex,:bpex]
#                    ,r=.9,rstep=0.0005,ry=0.01, seedξ=1,maxiters=2000);
