# alpha > alpha_d (I think TO CHECK)
E, σ = KSATBP.solve(N=10000, α=9.6, k=4, seed_cnf=19, r_step=0.0002, maxiters=1000);
@test E == 0

E, σ = KSATBP.solve(method=:decimation, N=10000,α=9.6, k=4, seed_cnf=19, r=0.02, maxiters=1000);
@test E == 0
