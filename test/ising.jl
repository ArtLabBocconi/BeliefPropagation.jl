using RRRMC
using FileIO

@testset "bp on erdos_renyi" begin
    Random.seed!(16)
    N, c = 100, 4
    net = erdos_renyi(N, c / N, Network)
    eprop!(net, "J", EdgeMap(net, e -> randn()))
    vprop!(net, "H", VertexMap(net, v -> 1 + randn()))
    fg = Ising.run_bp(net, T=2)

    @test mean(fg.mags) > 0.1
    # writenetwork("rrg.gml", net)
end

@testset "bp on rrg" begin
    T = 1
    N, z = 100, 4
    net = random_regular_graph(N, z, Network)
    eprop!(net, "J", EdgeMap(net, e -> 1))
    vprop!(net, "H", VertexMap(net, v -> 1 + randn()))
    fg = Ising.run_bp(net, T=T)

    X = rrg_to_mcgraph(net)
    Es, C, Cs = run_monte_carlo(X, β=1/T, infotime=100, sweeps=10^6);
    Cs = Cs[length(Cs)÷2:end] # consider half as burn-in
    mc_mags = 1 .- 2 .* reduce(+, Cs) / length(Cs)
    Δ = mean(abs, fg.mags .- mc_mags)
    @test Δ <= 0.02
end
