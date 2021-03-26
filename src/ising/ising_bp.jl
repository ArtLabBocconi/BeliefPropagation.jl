using ExtractMacro
using Printf

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

function FactorGraphIsingRRG(N::Int, k::Int, seed_graph::Int = -1)
    g = random_regular_graph(N, k, seed=seed_graph)
    adjlist = g.fadjlist
    @assert(length(adjlist) == N)
    vnodes = [VarIsing() for i=1:N]
    J = [Vector{T}() for i=1:N]

    for (i, v) in enumerate(vnodes)
        @assert(length(adjlist[i]) == k)
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
            v.tJ[ki] = 0
            J[i][ki] = 0
        end
    end

    mags = mag.(vnodes)

    FactorGraphIsing(N, vnodes, adjlist, J, mags)
end

function initrandJ!(g::FactorGraphIsing; μ=0, σ=1)
    for (i,v) in enumerate(g.vnodes)
        for (ki, j) in enumerate(g.adjlist[i])
            (i > j) && continue
            r = μ + σ * randn()
            g.J[i][ki] = r
            g.vnodes[i].tJ[ki] = tanh(r)
            kj = findfirst(==(i), g.adjlist[j])
            g.vnodes[j].tJ[kj] = g.vnodes[i].tJ[ki]
            g.J[j][kj] = g.J[i][ki]
        end
    end
end

function initrandH!(g::FactorGraphIsing; μ=0, σ=1)
    for v in g.vnodes
        v.H = μ + σ * randn()
    end
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
    corrs = Vector{Vector{Float64}}(g.N)
    for i=1:g.N
        corrs[i] = zeros(length(g.adjlist[i]))
    end

    for i=1:g.N
        for (ki,j) in enumerate(g.adjlist[i])
            corrs[i][ki] = corr_disc_nn(g,i,j)
        end
    end
    corrs
end

function setH!(g::FactorGraphIsing, H::Vector)
    @assert(g.N == length(H))
    for (i,v) in enumerate(g.vnodes)
        v.H = H[i]
    end
end

function setH!(g::FactorGraphIsing, H::Float64)
    for v in g.vnodes
        v.H = H
    end
end

function setMess!(g::FactorGraphIsing, H::Vector)
    @assert(g.N == length(H))
    for (i,v) in enumerate(g.vnodes)
        for k=1:deg(v)
            v.uin[k] = H[i]
        end
    end
end

function setJ!(g::FactorGraphIsing, J::Vector{Vector})
    @assert(g.N == length(J))
    for (i,v) in enumerate(g.vnodes)
        @assert(deg(v) == length(J[i]))
        for k=1:deg(v)
            v.tJ[k] = tanh(J[i][k])
            g.J[i][k] = J[i][k]
        end
    end
end

function setJ!(g::FactorGraphIsing, i::Int, j::Int, J)
    vi = g.vnodes[i]
    vj = g.vnodes[j]
    ki = findfirst(==(j), g.adjlist[i])
    kj = findfirst(==(i), g.adjlist[j])
    @assert(ki > 0)
    @assert(kj > 0)
    g.J[i][ki] = J
    g.J[j][kj] = J
    vi.tJ[ki] = tanh(J)
    vj.tJ[kj] = vi.tJ[ki]
end

function getJ(g::FactorGraphIsing, i::Int, j::Int)
    ki = findfirst(==(j), g.adjlist[i])
    g.J[i][ki]
end

function main_ising(; N::Int=1000, k::Int=4, 
                    β = 1., 
                    μJ=0, σJ=1,
                    μH=0, σH=1,
                    maxiters::Int=1000, ϵ=1e-6)
    g = FactorGraphIsingRRG(N, k)
    initrandJ!(g, μ=β*μJ, σ=β*σJ)
    initrandH!(g, μ=β*μH, σ=β*σH)
    initrandMess!(g, μ=0, σ=1)
    converge!(g, maxiters=maxiters, ϵ=ϵ)
    return g
end
