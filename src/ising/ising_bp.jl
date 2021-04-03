
mutable struct VarIsing
    uin::Vector{Float64}
    uout::Vector{Ptr{Float64}}
    tJ::Vector{Float64}
    H::Float64
    htot::Float64
end

VarIsing() = VarIsing(Vector{Float64}(), Vector{Ptr{Float64}}(), Vector{Float64}(), 0, 0)

mag(v::VarIsing) = tanh(v.htot)
deg(v::VarIsing) = length(v.uin)

function Base.show(io::IO, v::VarIsing)
    print(io, "VarIsing(deg=$(deg(v)), H=$(v.H))")
end

mutable struct FactorGraphIsing <: FactorGraph
    N::Int
    vnodes::Vector{VarIsing}
    adjlist::Vector{Vector{Int}}
    mags::Vector{Float64}
end

function FactorGraphIsing(net::Network; T=1)
    @assert has_eprop(net, "J") || has_gprop(net, "J")
    @assert has_vprop(net, "H") || has_gprop(net, "H")

    hasconstJ = !has_eprop(net, "J") 
    hasconstH = !has_vprop(net, "H")

    adjlist = adjacency_list(net)
    N = nv(net)
    vnodes = [VarIsing() for i=1:N]
    if hasconstJ
        J = gprop(net, "J") / T
    else
        J = [[eprop(net, e)["J"] / T for e in edges(net, i)] for i=1:N]
    end
    for (i, v) in enumerate(vnodes)
        resize!(v.uin, length(adjlist[i]))
        resize!(v.uout, length(adjlist[i]))
        resize!(v.tJ, length(adjlist[i]))
    end

    for (i, v) in enumerate(vnodes)
        for (ki, j) in enumerate(adjlist[i])
            v.uin[ki] = 0
            kj = findfirst(==(i), adjlist[j])
            vnodes[j].uout[kj] = getref(v.uin, ki)
            if hasconstJ
                v.tJ[ki] = tanh(J)
            else
                v.tJ[ki] = tanh(J[i][ki])
            end
        end
        if hasconstH
            v.H = gprop(net, "H") / T
        else
            v.H = vprop(net, i)["H"] / T
        end
        v.htot = sum(v.uin) + v.H
    end

    mags = mag.(vnodes)

    FactorGraphIsing(N, vnodes, adjlist, mags)
end

function initrand!(g::FactorGraphIsing; μ=0, σ=1)
    for v in g.vnodes
        for k=1:deg(v)
            v.uin[k] = μ + σ * randn()
        end
        v.htot = sum(v.uin) + v.H
    end
end

function getmess(g::FactorGraph, i, j)
    ki = findfirst(==(j), g.adjlist[i])
    return g.vnodes[i].uout[ki][]
end

function update!(v::VarIsing)
    @extract v: uin uout tJ
    v.htot = sum(uin) + v.H
    for k=1:deg(v)
        hcav = v.htot - uin[k]
        ucav = atanh(tJ[k] * tanh(hcav))
        uout[k][] = ucav
    end
end

function oneBPiter!(g::FactorGraphIsing)
    for i in randperm(g.N)
        update!(g.vnodes[i])
    end
    Δ = 0. 
    for i=1:g.N
        m = mag(g.vnodes[i])
        Δ = max(Δ, abs(g.mags[i] - m))
        g.mags[i] = m
    end
    return Δ
end

function converge!(g::FactorGraphIsing; maxiters=1000, ϵ=1e-6, verbose=true)
    for it=1:maxiters
        Δ = oneBPiter!(g)
        verbose && @printf("it=%d  Δ=%.2g \n", it, Δ)
        if Δ < ϵ
            verbose && println("Converged!")
            break
        end
    end
end

function corr_conn_nn(g::FactorGraphIsing, i::Int, j::Int)
    @extract g: N vnodes adjlist
    vi = vnodes[i]
    vj = vnodes[j]
    ki = findfirst(==(j), adjlist[i])
    kj = findfirst(==(i), adjlist[j])
    @assert(ki >0)
    @assert(kj >0)

    tJ = vi.tJ[ki]
    mij = tanh(htot(vi) - vi.uin[ki])
    mji = tanh(htot(vj) - vj.uin[kj])

    c = tJ * (1-mij^2)*(1-mji^2) / (1+tJ * mij * mji)
    return c
end

corr_disc_nn(g::FactorGraphIsing,i::Int,j::Int) = corr_conn_nn(g,i,j) + mag(g.vnodes[i])*mag(g.vnodes[j])

function corr_disc_nn(g::FactorGraphIsing)
    corrs = [zeros(length(g.adjlist[i])) for i=1:g.N]
    
    for i=1:g.N
        for (ki, j) in enumerate(g.adjlist[i])
            corrs[i][ki] = corr_disc_nn(g, i, j)
        end
    end
    return corrs
end

function run_bp(net::Network; 
                maxiters=1000, #max bp iters
                ϵ=1e-6, # stopping criterium 
                T=1,    # temperature
                μ=0,    # mean init messages
                σ=1,    # std init messages
                verbose=true,
                )
    g = FactorGraphIsing(net; T)
    initrand!(g; μ, σ)
    converge!(g; maxiters, ϵ, verbose)
    return g
end
