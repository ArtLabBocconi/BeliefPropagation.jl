include("../src/BeliefPropagation.jl")
using BeliefPropagation
using Base.Test

println("@ START TESTING")
println("@ Testing Functions...")
include("functions.jl")
println("@ ... done.")

println("@ Testing DeepLearning...")
include("deeplearning.jl")
println("@ ... done.")

println("@ Testing KSAT...")
include("ksat.jl")
println("@ ... done.")

println("@ ALL TESTS PASSED!")
