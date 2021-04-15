Random.seed!(17)
N = 100
net = CompleteBipartiteGraph(N, N, Network)
eprop!(net, "w", e -> rand())
vprop!(net, "b", v -> 1)
res_match = Matching.run_bp(net, maxiters=100, verbose=false)
res_bmatch = BMatching.run_bp(net, maxiters=100, verbose=false)
bmatchmap = [m[1] for m in res_bmatch.match]
@test all(bmatchmap .== res_match.match)

Random.seed!(17)
N = 100
net = CompleteGraph(N, Network)
eprop!(net, "w", e -> rand())
vprop!(net, "b", v -> rand(1:5))
res = BMatching.run_bp(net, maxiters=100, verbose=false)
@test res.num_violations <= 1 # = 0 on Julia 1.6
@test res.iters <= 100
