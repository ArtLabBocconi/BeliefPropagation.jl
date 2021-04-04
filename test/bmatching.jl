Random.seed!(17)
N = 100
net = CompleteBipartiteGraph(N, N, Network)
eprop!(net, "w", e -> rand())
vprop!(net, "b", v -> 1)
E, matchmap, g, nfails = Matching.run_bp(net, maxiters=100, verbose=false)
E, bmatchmap, g, nfails = BMatching.run_bp(net, maxiters=100, verbose=false)
bmatchmap = [m[1] for m in bmatchmap]
@test all(bmatchmap .== matchmap)

Random.seed!(17)
N = 100
net = CompleteGraph(N, Network)
eprop!(net, "w", e -> rand())
vprop!(net, "b", v -> rand(1:5))
E, σ, g, nfails = BMatching.run_bp(net, maxiters=100, verbose=false)
@test nfails == 0 # no violated constraints
