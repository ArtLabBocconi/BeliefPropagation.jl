### PERCEPTRON
@time g, W, E, stab = DeepBinary.solve(α=0.5, K=[201,1]
            , layers=[:ms]
            ,r=1.,r_step=0.0, seedξ=1,maxiters=1000);
@test E == 0

for lay in [:tapex,:bpex]
    @time g, W, E, stab = DeepBinary.solve(α=0.7, K=[101,1]
                , layers=[lay]
                ,r=.3,r_step=0.002, seedξ=1,maxiters=500);
    @test E == 0
end

for lay in [:tap,:bp]
    @time g, W, E, stab = DeepBinary.solve(α=0.7, K=[1001,1]
                , layers=[lay]
                ,r=.6,r_step=0.05, seedξ=1,maxiters=500);
    @test E == 0
end

#### COMMITTEE
for lay in [:tapex, :bpex]
    @time g, W, E, stab = DeepBinary.solve(α=0.2, K=[1001,7,1]
                , layers=[:tap,lay]
                ,r=.8,r_step=0.01, seedξ=1,maxiters=500);
    @test E == 0
end

### COMMITTEE CONTINUOUS FIRST LAYER
@time g, W, E, stab = DeepBinary.solve(K=[301,5,1]
                   , layers=[:bpreal,:bpex]
                   ,r=0.2,r_step=0.002, ry=0.2, altconv=true, altsolv=true,seedξ=1,maxiters=1000, plotinfo=0,β=Inf, α=2.,maketree=false);
@test E == 0

### 3 LAYERS
@time g, W, E, stab = DeepBinary.solve(α=0.2, K=[401,21,3,1]
            , layers=[:tap,:tapex,:tapex]
            ,r=.92,r_step=0., seedξ=1,maxiters=2000);
@test E == 0

@time g, W, E, stab = DeepBinary.solve(α=0.2, K=[401,21,3,1]
            , layers=[:tap,:bpex,:tapex]
            ,r=.95,r_step=0.001, seedξ=1,maxiters=2000);
@test E == 0

# too slow
# @time g, W, E, stab = DeepBinary.solve(α=0.2, K=[401,21,3,1]
#             , layers=[:tap,:bpex,:bpex]
#             ,r=.95,r_step=0.005, seedξ=1,maxiters=1000);


###########################
