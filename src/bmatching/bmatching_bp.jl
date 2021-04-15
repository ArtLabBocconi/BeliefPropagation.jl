mutable struct Fact
    uin::Vector{Float64}
    uout::Vector{Ptr{Float64}}
    neigs::Vector{Int}
    w::Vector{Float64}
    b::Int
end

Fact() = Fact(Float64[], Ptr{Float64}[], Int[], Float64[], 1)

deg(f::Fact) = length(f.uin)

mutable struct FGBMatching
    N::Int
    γ::Float64
    fnodes::Vector{Fact}
    adjlist::Vector{Vector{Int}}
end

show(io, ::FGBMatching) = show(io, "FGBMatching(...)")

function FGBMatching(net::Network; γ=Inf)
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

    FGBMatching(N, γ, fnodes, adjlist)
end

function initrand!(g::FGBMatching)
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
    
    Δ = abs(uout[is[b]][] - m2)
    for i=1:deg(f)
        uout[i][] = m1
    end
    for i in is[1:b]
        uout[i][] = m2
    end
    return Δ
end

findmatch(f::Fact) = findmatch(f, f.b)

function findmatch(f::Fact, k)
    h = f.w .- f.uin
    is = topk(h, k, rev=true)    
    return is
end

function oneBPiter!(g::FGBMatching)
    Δ = 0.
    for a in randperm(g.N)
        d = update!(g.fnodes[a])
        Δ = max(Δ, d)
    end
    return Δ
end

function converge!(g::FGBMatching; maxiters=100, ϵ=1e-8, verbose=true)
    Eold = 0.
    tstop = 0
    it = 0
    while it < maxiters
        it += 1
        Δ = oneBPiter!(g)
        E, matchmap, nviolations = energy(g)
        verbose && @printf("it=%d  E=%.5f  nviolations=%d \tΔ=%f \n", it, E, nviolations, Δ)
        
        if abs(Eold - E) < ϵ && nviolations == 0
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
    return Eold, it
end

function energy(g::FGBMatching)
    @extract g: fnodes N adjlist
    E = 0.
    matchmap = Vector{Vector{Int}}(undef, N) 
    for i=1:N
        f = fnodes[i]
        ks = findmatch(f)
        E += sum(f.w[ks])
        matchmap[i] = adjlist[i][ks]
    end
    E /= 2
    nviolations = 0
    for i=1:N
        for j in matchmap[i]
            nviolations += i ∉ matchmap[j]
        end
    end
    return E, matchmap, nviolations
end

"""
    run_bp(net::Network; [γ, maxiters, ϵ, seed, verbose])

Computes the minimum weight perfect b-matching on graph `net`.
The input `net` has to contain the edge property `w` and
the vertex property `b`.

Can optionally impose a cutoff `γ` on the edge costs `w`
in order to sparsify the graph and improve performance.

The output object contains the following fields:

- `energy`: the optimal cost.
- `match`: where `match[i]` is the list of neighbors of vertex 
           `i` in the optimal matching.
- `iters`: number of iterations performed.
- `ok`: whether the algorithm was successful or not.    
- `num_violations`: the number of inconsistencies in the solution. 
- `bpgraph`: a type storing the BP messages.
"""
function run_bp(net::Network; 
                γ = Inf,
                maxiters = 10000, 
                ϵ = 1e-4,
                seed = -1, 
                verbose=true)
    seed > 0 && Random.seed!(seed)
    g = FGBMatching(net; γ)
    initrand!(g)
    E, iters = converge!(g; maxiters, ϵ, verbose)
    E, matchmap, nviolations = energy(g)
    
    res = (energy = E,
           match = matchmap,
           num_violations =  nviolations,
           iters = iters,
           ok = (nviolations == 0),
           bpgraph = g)

    return res
end
