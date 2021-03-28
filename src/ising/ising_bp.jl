const T = Float64  

mutable struct VarIsing
    uin::Vector{T}
    uout::Vector{Ptr{T}}
    tJ::Vector{T}
    H::T
end

VarIsing() = VarIsing(Vector{T}(), Vector{Ptr{T}}(), Vector{T}(), 0)

htot(v::VarIsing) = sum(v.uin) + v.H
mag(v::VarIsing) = tanh(htot(v))
deg(v::VarIsing) = length(v.uin)

function Base.show(io::IO, v::VarIsing)
    print(io, "VarIsing(deg=$(deg(v)), H=$(v.H))")
end


abstract type FactorGraph end

mutable struct FactorGraphIsing <: FactorGraph
    N::Int
    vnodes::Vector{VarIsing}
    adjlist::Vector{Vector{Int}}
    J::Vector{Vector{T}}
    mags::Vector{T}
end

function FactorGraphIsing(net::Network; T=1)
    @assert has_eprop(net, "J")
    @assert has_vprop(net, "H")

    adjlist = adjacency_list(net)
    N = nv(net)
    vnodes = [VarIsing() for i=1:N]
    J = [[eprop(net, e)["J"] / T for e in edges(net, i)] for i=1:N]

    for (i, v) in enumerate(vnodes)
        v.H = vprop(net, i)["H"] / T
        resize!(v.uin, length(adjlist[i]))
        resize!(v.uout, length(adjlist[i]))
        resize!(v.tJ, length(adjlist[i]))
        resize!(J[i], length(adjlist[i]))
    end

    for (i, v) in enumerate(vnodes)
        for (ki, j) in enumerate(adjlist[i])
            v.uin[ki] = 0
            kj = findfirst(==(i), adjlist[j])
            vnodes[j].uout[kj] = getref(v.uin, ki)
            v.tJ[ki] = tanh(J[i][ki])
        end
    end

    mags = mag.(vnodes)

    FactorGraphIsing(N, vnodes, adjlist, J, mags)
end

function initrandMess!(g::FactorGraphIsing; μ=0, σ=1)
    for v in g.vnodes
        for k=1:deg(v)
            v.uin[k] = μ + σ * randn()
        end
    end
end

function update!(v::VarIsing)
    @extract v: uin uout tJ
    ht = htot(v)
    for k=1:deg(v)
        hcav = ht - uin[k]
        ucav = atanh(tJ[k] * tanh(hcav))
        uout[k][] = ucav
    end
end

function oneBPiter!(g::FactorGraphIsing)
    for i in randperm(g.N)
        update!(g.vnodes[i])
    end
    new_mags = mag.(g.vnodes)
    Δ = mean(abs, new_mags .- g.mags)
    g.mags .= new_mags    
    return Δ
end

function converge!(g::FactorGraphIsing; maxiters=1000, ϵ=1e-6, verbose=true)
    for it=1:maxiters
        verbose && print("it=$it ... ")
        Δ = oneBPiter!(g)
        verbose && @printf("Δ=%.2g \n", Δ)
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

function run_bp(net::Network; maxiters::Int=1000, ϵ=1e-6, T=1)
    g = FactorGraphIsing(net; T)
    initrandMess!(g, μ=0, σ=1)
    converge!(g; maxiters, ϵ)
    return g
end
