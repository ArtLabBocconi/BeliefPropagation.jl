
using Plots
using BeliefPropagation
using Erdos
using Random, Statistics

# 2D lattice

T = 2
L = 8
net = Grid([L, L], Network, periodic=true)
# Set J and H has graph properties since they are constants
gprop!(net, "T", T)
eprop!(net, "J", e -> rand()) 
vprop!(net, "H", v -> randn())

fg = Ising.run_bp(net, T=T, maxiters=1000, μ=1, verbose=true);

vprop!(net, "bp_mags", fg.mags)
eprop!(net, "bp_umess", e -> Ising.getmess(fg, src(e), dst(e)))
writenetwork("disferromag_2d_L$L.graphml", net)
