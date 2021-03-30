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
    ## NO COUPLINGS
    T = 3
    N, z = 100, 4
    Random.seed!(17)
    net = random_regular_graph(N, z, Network)
    eprop!(net, "J", e -> 0)
    vprop!(net, "H", v -> randn())
    fg = Ising.run_bp(net, T=T)

    @test fg.mags ≈ tanh.(vprop(net, "H").data ./ T)

    ## NO FIELDS, J=1
    T = 2
    N, z = 100, 4
    net = random_regular_graph(N, z, Network)
    eprop!(net, "J", e -> 1)
    vprop!(net, "H", v -> 0)
    fg = Ising.run_bp(net, T=T)

    # X = rrg_to_mcgraph(net)
    # Es, σ, mc_mags = run_monte_carlo(X, β=1/T, infotime=10, sweeps=10^7);
    # Δ = mean(abs, fg.mags .- mc_mags)
    # @test Δ <= 0.01
end
