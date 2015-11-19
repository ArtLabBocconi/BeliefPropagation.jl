using MacroUtils
include("cnf.jl")
typealias MessU Float64  # ̂ν(a→i) = P(σ_i != J_ai)
typealias MessH Float64 #  ν(i→a) = P(σ_i != J_ai)
MessU()= MessU(1.)

typealias PU Ptr{MessU}
typealias PH Ptr{MessH}
#getref(v::Vector, i::Integer) = pointer(v, i)
#getref{T}(v::Vector{T}, i::Integer) = Base.unsafe_convert(Ptr{T}, Base.data_pointer_from_objref(v[i]))

# typealias PU Ref{MessU}
# typealias PH Ref{MessH}
# getref(v::Vector, i::Integer) = Ref(v, i)


#Base.getindex(p::Ptr) = unsafe_load(p)
#Base.getindex{T}(p::Ptr{T}) = Base.unsafe_pointer_to_objref(p)::T
#Base.setindex!{T}(p::Ptr{T}, x::T) = unsafe_store!(p, x)
typealias VU Vector{MessU}
typealias VH Vector{MessH}
typealias VRU Vector{PU}
typealias VRH Vector{PH}

immutable Pi
    p::MessH
    m::MessH
    Pi() = new(1.0, 1.0)
    Pi(p, m) = new(p, m)
end

immutable Var
    π1::Pi

    #used only in BP+reinforcement
    π0::Pi
    η̄reinf::Pi

    Var() = new(Pi(), Pi(), Pi())
    Var(π1::Pi, π0::Pi, η̄reinf::Pi) = new(π1, π0, η̄reinf)
end

newvar1(v::Var, π1::Pi) = Var(π1, v.π0, v.η̄reinf)
newvar0(v::Var, π0::Pi) = Var(v.π1, π0, v.η̄reinf)
newvarη̄(v::Var, η̄reinf::Pi) = Var(v.π1, b.π0, η̄reinf)

immutable Literal
    η̄::MessU
    var::Int
    J::Int
end

newη̄(l::Literal, η̄) = Literal(η̄, l.var, l.J)

type Fact{K}
    lit::NTuple{K,Literal}
end
getK{K}(::Fact{K}) = K

abstract FactorGraph
type FactorGraphKSAT <: FactorGraph
    N::Int
    M::Int
    fnodes::Dict{Int,Vector{Fact}}
    vnodes::Vector{Var}
    σ::Vector{Int}
    cnf::CNF
    ζ::Vector{MessH}
    fperm::Vector{Int}

    function FactorGraphKSAT(cnf::CNF)
        @extract cnf M N clauses
        println("# read CNF formula")
        println("# N=$N M=$M α=$(M/N)")
        #fnodes = [Fact() for i=1:M]
        vnodes = [Var() for i=1:N]

        Ks = map(length, clauses)

        fnodes = Dict{Int,Vector{Fact}}()
        #sizehint!(fnodes, M)
        for (a,clause) in enumerate(clauses)
            K = Ks[a]
            haskey(fnodes, K) || (fnodes[K] = Fact{K}[])
            push!(fnodes[K],
                Fact{K}(ntuple(i->begin
                    id = clause[i]
                    @assert id ≠ 0
                    Literal(MessU(), abs(id), sign(id))
                end, K)))
        end
        maxK = maximum(keys(fnodes))

        σ = ones(Int, N)
        ζ = zeros(MessH, maxK)
        fperm = collect(1:M)
        new(N, M, fnodes, vnodes, σ, cnf, ζ, fperm)
    end
end

function getfnodes{K}(fnodes::Dict{Int,Vector{Fact}}, ::Type{Val{K}})
    fv = fnodes[K]
    return pointer_to_array(convert(Ptr{Fact{K}}, pointer(fv)), length(fv))::Vector{Fact{K}}
end

type ReinfParams
    reinf::Float64
    step::Float64
    wait_count::Int
    ReinfParams(reinf=0., step = 0.) = new(reinf, step, 0)
end

function initrand!(g::FactorGraphKSAT)
    for i = 1:length(g.vnodes)
        g.vnodes[i] = Var()
    end

    πp = ones(g.N)
    πm = ones(g.N)

    for (K,fv) in g.fnodes, f in fv
        newlit = ntuple(k->begin
                l = f.lit[k]
                r = 0.5 * rand()
                η̄ = r / (1 - r)
                vi = l.var
                if l.J == 1
                    πp[vi] *= η̄
                else
                    πm[vi] *= η̄
                end
                return newη̄(l, η̄)
            end, getK(f))
        f.lit = newlit
    end

    for (i,v) in enumerate(g.vnodes)
        g.vnodes[i] = newvar1(v, Pi(πp[i], πm[i]))
    end

    return g
end

function update1(vnodes, vi, η̄, J, ζprod, nzeros)
    eps = 1e-15
    π1 = vnodes[vi].π1
    πp = π1.p
    πm = π1.m
    if J == 1
        πu = πp / η̄
        πs = πm
    else
        πu = πm / η̄
        πs = πp
    end
    ζ = πu / (πu + πs)
    if ζ > eps
        ζprod *= ζ
    else
        nzeros += 1
    end
    return ζ, ζprod, nzeros
    #ζ[i] = z
end

function update2(l::Literal, vnodes::Vector{Var}, ζprod, nzeros, ζ)
    eps = 1e-15
    #l = lit[i]
    if nzeros == 0
        η = ζprod / ζ
    elseif nzeros == 1 && ζ < eps
        η = ζprod
    else
        η = 0.
    end
    η̄new = 1 - η
    η̄old = l.η̄
    vi = l.var
    v = vnodes[vi]
    πp = v.π1.p
    πm = v.π1.m
    if l.J == 1
        πp *= η̄new / η̄old
    else
        πm *= η̄new / η̄old
    end
    vnodes[vi] = newvar1(v, Pi(πp, πm))
    return newη̄(l, η̄new)
    # Δ = max(Δ, abs(ηi  - old))
end

@generated function newlit{K}(lit::NTuple{K,Literal}, vnodes::Vector{Var}, ζprod, nzeros, ζ)
    args = [:(update2(lit[$i], vnodes, ζprod, nzeros, ζ[$i])) for i = 1:K]
    return :(tuple($(args...)))
end

function update!{K}(f::Fact{K}, vnodes::Vector{Var}, ζ::Vector{MessH})
    @extract f lit
    Δ = 1.
    ζprod = 1.
    #eps = 1e-15
    nzeros = 0
    @inbounds for i = 1:K
        l = lit[i]
        ζ[i], ζprod, nzeros = update1(vnodes, l.var, l.η̄, l.J, ζprod, nzeros)
    end
    f.lit = newlit(lit, vnodes, ζprod, nzeros, ζ)
    return Δ
end

function update(v::Var, reinf::Float64 = 0.)

    #### update reinforcement ######
    πp = v.π1.p
    πm = v.π1.m

    πp0 = v.π0.p
    πm0 = v.π0.m

    η̄reinfp = v.η̄reinf.p
    η̄reinfm = v.η̄reinf.m

    πp /= η̄reinfp
    πm /= η̄reinfm

    if πp0 < πm0
        #η̄reinfp = (πp0 / πm0)^reinf
        #η̄reinfp = e^(reinf * (log(πp0) - log(πm0)))
        η̄reinfp = 1 - reinf * (πm0 - πp0) / (πm0 + πp0)
        η̄reinfm = 1.0
    else
        #η̄reinfm = (πm0 / πp0)^reinf
        #η̄reinfm = e^(reinf * (log(πm0) - log(πp0)))
        η̄reinfm = 1 - reinf * (πp0 - πm0) / (πm0 + πp0)
        η̄reinfp = 1.0
    end

    πp *= η̄reinfp
    πm *= η̄reinfm

    return Var(Pi(πp, πm), Pi(πp, πm), Pi(η̄reinfp, η̄reinfm))
end

function oneKiter!{K}(fnodes::Vector{Fact{K}}, vnodes::Vector{Var}, ζ::Vector{MessH})
    for f in shuffle(fnodes)
        d = update!(f, vnodes, ζ)
    end
end

function oneBPiter!(g::FactorGraph, reinf::Float64=0.)
    @extract g fnodes vnodes ζ
    Δ = 1.

    for K in keys(fnodes)
        fv = getfnodes(fnodes, Val{K})
        oneKiter!(fv, vnodes, ζ)
    end

    #shuffle!(fperm)
    #for a in fperm
        #d = update!(fnodes[a], vnodes)
        #Δ = max(Δ, d)
    #end

    for i = 1:length(g.vnodes)
        g.vnodes[i] = update(g.vnodes[i], reinf)
    end

    return Δ
end

function update_reinforcement!(reinfpar::ReinfParams)
    if reinfpar.wait_count < 4
        reinfpar.wait_count += 1
    else
        reinfpar.reinf = 1 - (1-reinfpar.reinf) * (1-reinfpar.step)
    end
end

function getconfig!(g::FactorGraph)
    @extract g N vnodes σ
    @inbounds for i = 1:N
        π1 = vnodes[i].π1
        σ[i] = 1 - 2 * (π1.p > π1.m)
    end
    return σ
end

function converge!(g::FactorGraph; maxiters::Int = 100, ϵ::Float64=1e-5, reinfpar::ReinfParams=ReinfParams())
    for it=1:maxiters
        @printf("it=%i ... ", it)
        Δ = oneBPiter!(g, reinfpar.reinf)
        σ = getconfig!(g)
        E = energy(g.cnf, σ)
        @printf("reinf=%.3f E=%d  Δ=%f\n", reinfpar.reinf, E, Δ)
        update_reinforcement!(reinfpar)
        if E == 0
            println("Found Solution!")
            break
        end
        if Δ < ϵ
            println("Converged!")
            break
        end
    end
end

function energy(cnf::CNF, σ)
    @extract cnf M clauses
    E = M
    @inbounds for c in clauses
        for i in c
            if sign(i) == σ[abs(i)]
                E -= 1
                break
            end
        end
    end
    return E
end

function solveKSAT(cnfname::AbstractString; kw...)
    cnf = readcnf(cnfname)
    solveKSAT(cnf; kw...)
end

function solveKSAT(; N::Int=1000, α::Float64=3., k::Int = 4, seed_cnf::Int=-1, kw...)
    if seed_cnf > 0
        srand(seed_cnf)
    end
    cnf = CNF(N, k, α)
    solveKSAT(cnf; kw...)
end

function solveKSAT(cnf::CNF; maxiters::Int = 10000, ϵ::Float64 = 1e-6,
                reinf::Float64 = 0., reinf_step::Float64= 0.01,
                seed::Int = -1)
    if seed > 0
        srand(seed)
    end
    reinfpar = ReinfParams(reinf, reinf_step)
    g = FactorGraphKSAT(cnf)
    initrand!(g)
    converge!(g, maxiters=maxiters, ϵ=ϵ, reinfpar=reinfpar)
    return g
end
