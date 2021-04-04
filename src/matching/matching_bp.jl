mutable struct Fact
    uin::Vector{Float64}
    uout::Vector{Ptr{Float64}}
    neigs::Vector{Int}
    w::Vector{Float64}
end

Fact() = Fact(Float64[], Ptr{Float64}[], Int[], Float64[])

deg(f::Fact) = length(f.uin)

mutable struct FactorGraph
    N::Int
    γ::Float64
    fnodes::Vector{Fact}
    adjlist::Vector{Vector{Int}}
end

function FactorGraph(net::Network; γ=Inf)
    @assert has_eprop(net, "w")

    N = nv(net)
    fnodes = [Fact() for i=1:N]

    # Prune Graph
    w = [Float64[] for i=1:N]
    adjlist = [Int[] for i=1:N]
    for i=1:N
        for e in edges(net, i)
            wij = eprop(net, e)["w"]
            if wij < γ
                push!(w[i], wij)
                push!(adjlist[i], dst(e))
            end
        end
    end

    for (i, v) in enumerate(fnodes)
        resize!(v.uin, length(adjlist[i]))
        resize!(v.uout, length(adjlist[i]))
        resize!(v.w, length(adjlist[i]))
    end

    for (i, f) in enumerate(fnodes)
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

function update!(f::Fact)
    @extract f: w uin uout
    m1 = m2 = Inf
    i1 = 0
    for i=1:deg(f)
        h = w[i] - uin[i]
        if h < m1
            m2 = m1
            m1 = h
            i1 = i
        elseif h < m2
            m2 = h
        end
    end

    for i=1:deg(f)
        uout[i][] = m1
    end
    Δ = abs(uout[i1][] - m2)
    uout[i1][] = m2
    return Δ
end

function findmatch(f::Fact)
    @extract f: w uin
    m1 = Inf
    i1 = 0
    for i=1:deg(f)
        h = w[i] - uin[i]
        if h < m1
            m1 = h
            i1 = i
        end
    end
    return i1, w[i1]
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
    matchmap = zeros(Int, N) 
    for i=1:N
        f = fnodes[i]
        k, wij = findmatch(f)
        E += wij
        matchmap[i] = adjlist[i][k]
    end
    E /= 2
    nfails = 0
    for i=1:N
        j = matchmap[i]
        nfails += matchmap[j] != i
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
