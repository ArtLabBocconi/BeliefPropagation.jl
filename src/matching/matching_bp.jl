using MacroUtils
typealias MessU Float64  # ̂ν(a→i) = P(σ_i != J_ai)
typealias MessH Float64 #  ν(i→a) = P(σ_i != J_ai)

getref(v::Vector, i::Integer) = pointer(v, i)
MessU(x=0.) = x
MessH(x=0.) = x

typealias PU Ptr{MessU}
typealias PH Ptr{MessH}

typealias VU Vector{MessU}
typealias VH Vector{MessH}
typealias VRU Vector{PU}
typealias VRH Vector{PH}

type Fact
    h::VH
    u::VRU
    neigs::Vector{Int}
end

Fact() = Fact(VH(), VRU(), Int[])

type Var
    E::Float64
    u::VU
    h::VRH
    htot::MessH
    neigs::Vector{Int}
end

Var() = Var(0., VU(), VRH(), MessH(), Int[])


"""
Multi-index matching on d-partite hypergraph with
N*d nodes and N*γ hyperedges (factors and variables respectively)
"""
type FactorGraph
    N::Int
    d::Int
    γ::Float64
    β::Float64
    fnodes::Vector{Fact}
    vnodes::Vector{Var}

    function FactorGraph(N::Int, d::Int, γ::Float64; β=Inf)
        nf = N*d
        nv = round(Int, N * 2γ)
        # nv = N^d
        fnodes = [Fact() for i=1:nf]
        vnodes = [Var() for i=1:nv]
        fit = rand(nf) # case 1
        dims = tuple([N for _=1:d]...)
        for (i, v) in enumerate(vnodes)
            # v.E = 1. # case 1
            v.E = rand()*(2γ) # case 2
            # v.E = rand()*N^(d-1)
            # idx = ind2sub(dims, i)
            for l=1:d
                # a = N*(l-1) + idx[l]
                a = N*(l-1) + rand(1:N)
                f = fnodes[a]
                push!(f.neigs, i)
                push!(v.neigs, a)
                # v.E *= fit[a] # case 1
            end
            # v.E *= 2γ # case 1
        end

        ## ATTENTION: Reserve memory in order to avoid invalidation of Refs
        ## The degree of each node has to be exactly known
        for (a,f) in enumerate(fnodes)
            sizehint!(f.u, length(f.neigs))
            sizehint!(f.h, length(f.neigs))
        end
        for (i,v) in enumerate(vnodes)
            sizehint!(v.u, d)
            sizehint!(v.h, d)
        end

        for (i, v) in enumerate(vnodes)
            # idx = ind2sub(dims, i)
            for l=1:d
                a = v.neigs[l]
                # a = N*(l-1) + idx[l]
                f = fnodes[a]
                push!(v.u, MessU())
                push!(f.u, getref(v.u, length(v.u)))

                push!(f.h, MessH())
                push!(v.h, getref(f.h, length(f.h)))
            end
        end
        new(N, d, γ, β, fnodes, vnodes)
    end
end

type ReinfParams
    r::Float64
    rstep::Float64
    wait_count::Int
    ReinfParams(r=0.,rstep=0) = new(r, rstep, 0)
end

deg(f::Fact) = length(f.u)
deg(v::Var) = length(v.h)

function initrand!(g::FactorGraph)
    for f in g.fnodes
        for k=1:deg(f)
            f.h[k] = rand()
        end
    end
    for v in g.vnodes
        for k=1:deg(v)
            v.u[k] = rand()
        end
    end
end

function update!(f::Fact, γ, β = Inf)
    @extract f u h
    # m1 = m2 = γ
    if β == Inf
        m1 = m2 = γ
        i1 = 0
        for i=1:deg(f)
            if h[i] < m1
                m2 = m1
                m1 = h[i]
                i1 = i
            elseif h[i] < m2
                m2 = h[i]
            end
        end
        @assert i1 != 0
        @assert isfinite(m1)
        @assert isfinite(m2)
        for i=1:deg(f)
            u[i][] = m1
        end
        u[i1][] = m2
    else # β != Inf
        #TODO
    end
end


function update!(v::Var, r::Float64 = 0.)
    @extract v: u h
    Δ = 0.
    ### compute cavity fields
    htot = v.E
    for i=1:deg(v)
        htot -= u[i]
    end

    for i=1:deg(v)
        hold = h[i][]
        h[i][] = htot + u[i]
        Δ = max(Δ, abs(hold - h[i][]))
    end
    # Δ = max(Δ, abs(v.htot - htot))
    v.htot = htot

    Δ
end

function oneBPiter!(g::FactorGraph, r::Float64=0.)
    @extract g: fnodes vnodes N
    Δ = 0.
    nv = length(vnodes);
    nf = length(fnodes);

    for a=1:nf
        update!(g.fnodes[a], g.γ)
    end
    for i=1:nv
        d = update!(g.vnodes[i], r)
        Δ = max(Δ, d)
    end

    # for _=1:(nv+nf)
    #     n = rand(1:(nv+nf))
    #     if n <= nv
    #         d = update!(g.vnodes[n], r)
    #         Δ = max(Δ, d)
    #     else
    #         update!(g.fnodes[n-nv], g.γ)
    #     end
    # end

    Δ
end

function update_reinforcement!(reinfpar::ReinfParams)
    if reinfpar.wait_count < 10
        reinfpar.wait_count += 1
    else
        reinfpar.r *= 1 + reinfpar.rstep
    end
end

function converge!(g::FactorGraph; maxiters::Int = 100, ϵ::Float64=1e-5
        , reinfpar::ReinfParams=ReinfParams())

    for it=1:maxiters
        print("it=$it ... ")
        Δ = oneBPiter!(g, reinfpar.r)
        E, matchmap = energy(g)
        @printf("r=%.3f E=%.3f   \tΔ=%f \n", reinfpar.r, E, Δ)
        update_reinforcement!(reinfpar)

        if Δ < ϵ
            println("Converged!")
            break
        end
    end
end

function energy(g::FactorGraph; checkvalidity = false)
    @extract g: fnodes vnodes N
    E = 0.
    matchmap = zeros(Int, N) #only for d=2
    matchcount = zeros(Int, N)
    for a=1:N
        f = fnodes[a]
        i1 = 0
        m1 = 10000.
        for i=1:deg(f)
            if f.h[i] < m1
                m1 = f.h[i]
                i1 = i
            end
        end
        @assert i1 > 0
        v = vnodes[f.neigs[i1]]
        E += v.E
        matchmap[a] = v.neigs[2] - N
        matchcount[v.neigs[2] - N] += 1
    end
    if checkvalidity
        for i=1:N
            @assert matchcount[i] == 1
        end
    end
    # println("nv=$(nv/N) E=$(E/N)")
    return E / N, matchmap
end

"""
Return the optimal cost density and the matching map:
`matchmap[i] = j`  if (i,j) is in the optimal matching.

The cutoff on the costs is  2γ.
"""
function solve(;N=200, d=2, γ=40.,
                maxiters::Int = 10000, ϵ::Float64 = 1e-4,
                r::Float64 = 0., rstep::Float64= 0.00,
                seed::Int = -1)
    seed > 0 && srand(seed)
    g = FactorGraph(N, d, γ)
    initrand!(g)
    reinfpar = ReinfParams(r, rstep)
    converge!(g, maxiters=maxiters, ϵ=ϵ, reinfpar=reinfpar)
    E, matchmap = energy(g, checkvalidity = true)
    return E, matchmap
end
