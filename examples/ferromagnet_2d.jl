
using Plots
using BeliefPropagation
using Erdos
using Random, Statistics

# 2D lattice

L = 8
net = Grid([L, L], Network, periodic=true)
# Set J and H has graph properties since they are constants
gprop!(net, "J", 1) 
gprop!(net, "H", 0)
writenetwork("ferromagnet_2d_L$L.graphml", net)

Ts =  1:0.1:4
Ms = Float64[]
for T in Ts
    fg = Ising.run_bp(net, T=T, maxiters=1000, μ=1, verbose=false);
    m = abs(mean(fg.mags))
    push!(Ms, m)
end
Tc = 2 / log(1+√2) #2.269185314213022 from Onsager's solution
plot(Ts, Ms, label="BP")
title!("2D ferromagnet")
xlabel!("T")
ylabel!("M")
vline!([Tc], label="Tc Onsager", color="black")
