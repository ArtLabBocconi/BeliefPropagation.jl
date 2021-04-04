mutable struct Fact
    uin::Vector{Float64}
    uout::Vector{Ptr{Float64}}
    neigs::Vector{Int}
    w::Vector{Float64}
    b::Int
end

Fact() = Fact(Float64[], Ptr{Float64}[], Int[], Float64[], 1)

deg(f::Fact) = length(f.uin)

mutable struct FactorGraph
    N::Int
    γ::Float64
    fnodes::Vector{Fact}
    adjlist::Vector{Vector{Int}}
end

function FactorGraph(net::Network; γ=Inf)
    @assert has_eprop(net, "w")
    @assert has_vprop(net, "b")

    N = nv(net)
    fnodes = [Fact() for i=1:N]
    wmap = eprop(net, "w")
    bmap = vprop(net, "b")

    # Prune Graph
    w = [Float64[] for i=1:N]
    adjlist = [Int[] for i=1:N]
    for i=1:N
        for e in edges(net, i)
            wij = wmap[e]
            if wij < γ
                push!(w[i], wij)
                push!(adjlist[i], dst(e))
            end
        end
    end

    for (i, f) in enumerate(fnodes)
        resize!(f.uin, length(adjlist[i]))
        resize!(f.uout, length(adjlist[i]))
        resize!(f.w, length(adjlist[i]))
    end

    for (i, f) in enumerate(fnodes)
        f.b = bmap[i]
        for (ki, j) in enumerate(adjlist[i])
            f.uin[ki] = 0
            kj = findfirst(==(i), adjlist[j])
            fnodes[j].uout[kj] = getref(f.uin, ki)
            f.w[ki] = w[i][ki]
        end
    end

    FactorGraph(N, γ, fnodes, adjlist)
end

function initrand!(g::FactorGraph)
    for f in g.fnodes
        for k=1:deg(f)
            f.uin[k] = randn()
        end
    end
end

"""
    topk(x, k; rev=false)

Return the indexes of the largest `k` elements in `x`.
If `rev=true`, return the smallest elements.
"""
function topk(x, k; rev=false)
    is = partialsortperm(x, 1:k; rev=!rev)
    return is # xs[is]
end

function update!(f::Fact)
    @extract f: w uin uout b
    h = w .- uin
    is = topk(h, b+1, rev=true)
    m1 = h[is[b]]
    m2 = h[is[b+1]]
    for i=1:deg(f)
        uout[i][] = m1
    end
    Δ = abs(uout[is[b]][] - m2)
    for i in is[1:b]
        uout[i][] = m2
    end
    return Δ
end

function findmatch(f::Fact)
    @extract f: w uin b
    h = w .- uin
    is = topk(h, b, rev=true)    
    return is, sum(w[is])
end

function oneBPiter!(g::FactorGraph)
    Δ = 0.
    for a in randperm(g.N)
        d = update!(g.fnodes[a])
        Δ = max(Δ, d)
    end
    return Δ
end

function converge!(g::FactorGraph; maxiters=100, ϵ=1e-8, verbose=true)

    Eold = 0.
    tstop = 0
    
    for it=1:maxiters
        Δ = oneBPiter!(g)
        E, matchmap, nfails = energy(g)
        verbose && @printf("it=%d  E=%.5f  nfails=%d \tΔ=%f \n", it, E, nfails, Δ)
        
        if abs(Eold - E) < ϵ && nfails == 0
            tstop += 1
            if tstop == 10
                verbose && println("Found ground state")
                break
            end
        else
            tstop = 0
        end

        Eold = E
    end
    return Eold
end

function energy(g::FactorGraph)
    @extract g: fnodes N adjlist
    E = 0.
    matchmap = Vector{Vector{Int}}(undef, N) 
    for i=1:N
        f = fnodes[i]
        ks, sumw = findmatch(f)
        E += sumw
        matchmap[i] = adjlist[i][ks]
    end
    E /= 2
    nfails = 0
    for i=1:N
        for j in matchmap[i]
            nfails += i ∉ matchmap[j]
        end
    end
    return E, matchmap, nfails
end

"""
Return the optimal cost and the matching map:
`matchmap[i] = j`  if (i,j) is in the optimal matching.

The cutoff on the costs is  γ.
"""
function run_bp(net::Network; 
                γ = Inf,
                maxiters = 10000, 
                ϵ = 1e-4,
                seed = -1, 
                verbose=true)
    seed > 0 && Random.seed!(seed)
    g = FactorGraph(net; γ)
    initrand!(g)
    converge!(g; maxiters, ϵ, verbose)
    E, matchmap, nfails = energy(g)
    return E, matchmap, g, nfails
end
