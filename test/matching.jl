
E, σ = Matching.solve(N=500, γ=50., maxiters=2000);
@test abs(E - 3.14^2/6) < 0.1
