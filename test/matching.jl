Random.seed!(17)
N = 1000
net = CompleteBipartiteGraph(N, N, Network)
eprop!(net, "w", e -> rand())
res = Matching.run_bp(net, γ=0.1, maxiters=1000, verbose=false)
@test abs(res.energy - π^2/6) < 0.1
@test res.num_violations == 0 # no violated constraints

N = 1000
net = CompleteGraph(N, Network)
eprop!(net, "w", e -> rand())
res = Matching.run_bp(net, γ=0.1, maxiters=1000, verbose=false)
@test abs(res.energy - π^2/12) < 0.1
@test res.num_violations <= 5 # if graph is non-bipartite may fail
