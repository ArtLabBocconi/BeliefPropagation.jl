g = main_ising(; N=1000, k=4, 
                β = 1., 
                μJ=0, σJ=1,
                μH=1, σH=0,
                maxiters=100, ϵ=1e-6)

@test mean(g.mags) > 0
