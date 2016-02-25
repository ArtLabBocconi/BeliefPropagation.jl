### PERCEPTRON
@time g, W, E, stab = DeepBinary.solve(α=0.5, K=[201,1]
            , layers=[:ms]
            ,r=1.,r_step=0.0, seed_ξ=1,maxiters=1000);
@assert E == 0

for lay in [:tapex,:bpex]
    @time g, W, E, stab = DeepBinary.solve(α=0.7, K=[101,1]
                , layers=[lay]
                ,r=.3,r_step=0.002, seed_ξ=1,maxiters=500);
    @assert E == 0
end

for lay in [:tap,:bp]
    @time g, W, E, stab = DeepBinary.solve(α=0.7, K=[1001,1]
                , layers=[lay]
                ,r=.5,r_step=0.01, seed_ξ=1,maxiters=500);
    @assert E == 0
end

#### COMMITTEE
for lay in [:tapex,:bpex]
    @time g, W, E, stab = DeepBinary.solve(α=0.2, K=[1001,7,1]
                , layers=[:tap,lay]
                ,r=.8,r_step=0.01, seed_ξ=1,maxiters=500);

    @assert E == 0
end

### 3 LAYERS
@time g, W, E, stab = DeepBinary.solve(α=0.2, K=[401,21,3,1]
            , layers=[:tap,:tapex,:tapex]
            ,r=.9,r_step=0., seed_ξ=1,maxiters=1000);

@time g, W, E, stab = DeepBinary.solve(α=0.2, K=[401,21,3,1]
            , layers=[:tap,:bpex,:tapex]
            ,r=.95,r_step=0.001, seed_ξ=1,maxiters=1000);

@assert E == 0

###########################
