##### PERCEPTRON
for lay in [:tap,:bp]
    @time g, W, E, stab = DeepBinary.solve(α=0.7, K=[1001,1]
    , layers=[lay]
    ,r=.2,rstep=0.01, seedξ=1,maxiters=500);
    @test E == 0
end

@time g, W, E, stab = DeepBinary.solve(α=0.5, K=[201,1]
            , layers=[:ms]
            ,r=1.,rstep=0.0, seedξ=1,maxiters=1000);
@test E == 0

for lay in [:tapex,:bpex]
    @time g, W, E, stab = DeepBinary.solve(α=0.7, K=[101,1]
                , layers=[lay]
                ,r=.3,rstep=0.002, seedξ=1,maxiters=500);
    @test E == 0
end


### COMMITTEE
for lay in [:bpex] #TODO
    @time g, W, E, stab = DeepBinary.solve(α=0.2, K=[1001,7,1]
                , layers=[:tap,lay]
                ,r=.8,rstep=0.01, ry=0.3, seedξ=1,maxiters=500);
    @test E == 0
end
#
# ### COMMITTEE CONTINUOUS FIRST LAYER
# @time g, W, E, stab = DeepBinary.solve(K=[301,5,1] ,layers=[:bpreal,:bpex]
#                    ,r=0.2,rstep=0.002, ry=0.2, altconv=true, altsolv=true, seedξ=1,
#                    maxiters=1000, plotinfo=0,β=Inf, α=2.,maketree=false);
# @test E == 0
#
# ## 3 LAYERS

# @time g, W, E, stab = DeepBinary.solve(α=0.2, K=[401,21,3,1]
#             , layers=[:tap,:bp,:bp]
#             ,r=.92,rstep=0., seedξ=1,maxiters=2000);
# @test E == 0

# @time g, W, E, stab = DeepBinary.solve(α=0.2, K=[401,21,3,1]
#             , layers=[:tap,:tapex,:tapex]
#             ,r=.92,rstep=0., seedξ=1,maxiters=2000);
# @test E == 0
#
# @time g, W, E, stab = DeepBinary.solve(α=0.2, K=[401,21,3,1]
#             , layers=[:tap,:bpex,:tapex]
#             ,r=.95,rstep=0.001, seedξ=1,maxiters=2000);
# @test E == 0

#### too slow
####  @time g, W, E, stab = DeepBinary.solve(α=0.2, K=[401,21,3,1]
####              , layers=[:tap,:bpex,:bpex]
####             ,r=.95,rstep=0.005, seedξ=1,maxiters=1000);
####

# IMPARATO TERZO LIVELLO!!!!
# @time g, W, E, stab = DeepBinary.solve(α=0.15, K=[301,21,11,3,1]
#                 , layers=[:tap,:tap,:tapex,:bpex]
#                 ,r=.9, rstep=0.0005, ry=0.01, seedξ=1, maxiters=2000);

##########################
