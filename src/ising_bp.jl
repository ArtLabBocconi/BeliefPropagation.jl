using MacroUtils

typealias MessU Float64  
typealias PU Ptr{MessU}
typealias VU Vector{MessU}
typealias VPU Vector{PU}

type VarIsing
    uin::Vector{MessU}
    uout::Vector{PU}
    tJ::Vector{Float64}
    H::Float64
end

VarIsing() = VarIsing(VU(),VPU(), Vector{Float64}(), 0.)

abstract FactorGraph
type FactorGraphIsing <: FactorGraph
    N::Int
    vnodes::Vector{VarIsing}
    adjlist::Vector{Vector{Int}}
    J::Vector{Vector{Float64}}
end

function FactorGraphIsingRRG(N::Int, k::Int, seed_graph::Int = -1)
    g = random_regular_graph(N, k, seed=seed_graph)
    adjlist = g.fadjlist
    assert(length(adjlist) == N)
    vnodes = [VarIsing() for i=1:N]
    J = [Vector{Float64}() for i=1:N]

    for (i,v) in enumerate(vnodes)
        assert(length(adjlist[i]) == k)
        resize!(v.uin, length(adjlist[i]))
        resize!(v.uout, length(adjlist[i]))
        resize!(v.tJ, length(adjlist[i]))
        resize!(v.tJ, length(adjlist[i]))
        resize!(J[i], length(adjlist[i]))
    end

    for (i,v) in enumerate(vnodes)
        for (ki,j) in enumerate(adjlist[i])
            v.uin[ki] = MessU(i)
            kj = findfirst(adjlist[j], i)
            vnodes[j].uout[kj] = getref(v.uin, ki)
            v.tJ[ki] = 0.
            J[i][ki] = 0.
        end
    end

    FactorGraphIsing(N, vnodes, adjlist, J)
end

deg(v::VarIsing) = length(v.uin)


function initrandJ!(g::FactorGraphIsing; m::Float64=0., σ::Float64=1.)
    for (i,v) in enumerate(g.vnodes)
        for (ki, j) in enumerate(g.adjlist[i])
            (i > j) && continue
            r = m + σ * (rand() - 0.5)
            g.J[i][ki] = r
            g.vnodes[i].tJ[ki] = tanh(r)
            kj = findfirst(g.adjlist[j], i)
            g.vnodes[j].tJ[kj] = g.vnodes[i].tJ[ki]
            g.J[j][kj] = g.J[i][ki]
        end
    end
end

function initrandH!(g::FactorGraphIsing; m::Float64=0., σ::Float64=1.)
    for v in g.vnodes
        v.H = m + σ * (rand() - 0.5)
    end
end

function initrandMess!(g::FactorGraphIsing; m::Float64=1., σ::Float64=1.)
    for v in g.vnodes
        for k=1:deg(v)
            v.uin[k] = m + σ * (rand() - 0.5)
        end
    end
end

function update!(v::VarIsing)
    @extract v uin uout tJ
    Δ = 0.
    ht = htot(v)
    for k=1:deg(v)
        hcav = ht - uin[k]
        ucav = atanh(tJ[k]*tanh(hcav))
        Δ = max(Δ, abs(ucav  - uout[k][]))
        uout[k][] = ucav
    end
    Δ
end

function oneBPiter!(g::FactorGraphIsing)
    Δ = 0.
    for i=randperm(g.N)
        d = update!(g.vnodes[i])
        Δ = max(Δ, d)
    end
    Δ
end

function converge!(g::FactorGraphIsing; maxiters::Int = 1000, ϵ::Float64=1e-6, verbose::Bool=false)
    for it=1:maxiters
        if verbose
            write("it=$it ... ")
        end
        Δ = oneBPiter!(g)
        if verbose
            @printf("Δ=%f \n", Δ)
        end
        if Δ < ϵ
            if verbose
                println("Converged!")
            end
            break
        end
    end
end

htot(v::VarIsing) = sum(v.uin) + v.H
mag(v::VarIsing) = tanh(htot(v))
mags(g::FactorGraphIsing) = Float64[mag(v) for v in g.vnodes]

function corr_conn_nn(g::FactorGraphIsing, i::Int, j::Int)
    @extract g N vnodes adjlist
    vi = vnodes[i]
    vj = vnodes[j]
    ki = findfirst(adjlist[i], j)
    kj = findfirst(adjlist[j], i)
    assert(ki >0)
    assert(kj >0)

    tJ = vi.tJ[ki]
    mij = tanh(htot(vi) - vi.uin[ki])
    mji = tanh(htot(vj) - vj.uin[kj])

    c = tJ * (1-mij^2)*(1-mji^2) / (1+tJ * mij * mji)
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
    assert(g.N == length(H))
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
    assert(g.N == length(H))
    for (i,v) in enumerate(g.vnodes)
        for k=1:deg(v)
            v.uin[k] = H[i]
        end
    end
end

function setJ!(g::FactorGraphIsing, J::Vector{Vector})
    assert(g.N == length(J))
    for (i,v) in enumerate(g.vnodes)
        assert(deg(v) == length(J[i]))
        for k=1:deg(v)
            v.tJ[k] = tanh(J[i][k])
            g.J[i][k] = J[i][k]
        end
    end
end

function setJ!(g::FactorGraphIsing, i::Int, j::Int, J)
    vi = g.vnodes[i]
    vj = g.vnodes[j]
    ki = findfirst(g.adjlist[i],j)
    kj = findfirst(g.adjlist[j],i)
    assert(ki >0)
    assert(kj >0)
    g.J[i][ki] = J
    g.J[j][kj] = J
    vi.tJ[ki] = tanh(J)
    vj.tJ[kj] = vi.tJ[ki]

end

function getJ(g::FactorGraphIsing, i::Int, j::Int)
    ki = findfirst(g.adjlist[i],j)
    g.J[i][ki]
end

function mainIsing(; N::Int = 1000, k::Int = 4, β::Float64 = 1., maxiters::Int = 1000, ϵ::Float64=1e-6)
    g = FactorGraphIsingRRG(N, k)
    initrandJ!(g, m=β, σ=0.)
    initrandH!(g, m=0., σ=0.)
    initrandMess!(g, m=1., σ=1.)
    converge!(g, maxiters=maxiters, ϵ=ϵ)
    return mean(mags(g))
end
