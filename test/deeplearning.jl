# PERCEPTRON
# g, W, E, stab = DeepBinary.solve(α=0.7, K=[101,1], layers=[:tapex]
# ,r=.5,r_step=0.005, seed_ξ=1,maxiters=500);
# @assert E == 0

for lay in [:tap,:bp]
    g, W, E, stab = DeepBinary.solve(α=0.5, K=[1001,1], layers=[lay]
            ,r=.5,r_step=0.005, seed_ξ=1,maxiters=500);
    @assert E == 0
end

# COMMITTEE
g, W, E, stab = DeepBinary.solve(α=0.2, K=[1001,7,1], layers=[:tap,:tapex]
        ,r=.8,r_step=0.005, seed_ξ=1,maxiters=500);

@assert E == 0
