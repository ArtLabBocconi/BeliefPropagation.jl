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

type Var
    πp::MessH
    πm::MessH

    #used only in BP+reinforcement
    πp0::MessH
    πm0::MessH
    η̄reinfp::MessU
    η̄reinfm::MessU
    Var() = reset!(new())
end

function reset!(v::Var)
    v.πp = 1.0
    v.πm = 1.0
    v.πp0 = 1.0
    v.πm0 = 1.0
    v.η̄reinfm = 1.0
    v.η̄reinfp = 1.0
    return v
end

type Fact
    K::Int
    J::Vector{Int}
    η̄list::VU
    vlist::Vector{Var}
    ζ::VH
end
Fact() = Fact(0, Vector{Int}(), VU(), Vector{Var}(), VH())


abstract FactorGraph
type FactorGraphKSAT <: FactorGraph
    N::Int
    M::Int
    fnodes::Vector{Fact}
    vnodes::Vector{Var}
    σ::Vector{Int}
    cnf::CNF
    fperm::Vector{Int}

    function FactorGraphKSAT(cnf::CNF)
        @extract cnf M N clauses
        println("# read CNF formula")
        println("# N=$N M=$M α=$(M/N)")
        fnodes = [Fact() for i=1:M]
        vnodes = [Var() for i=1:N]

        Js = Vector{Int}[Int[sign(id) for id in clause] for clause in clauses]
        kf = map(length, clauses)

        ## Reserve memory in order to avoid invalidation of Refs
        for (a,f) in enumerate(fnodes)
            sizehint!(f.η̄list, kf[a])
            sizehint!(f.vlist, kf[a])
            resize!(f.ζ, kf[a])
        end

        for (a,clause) in enumerate(clauses)
            f = fnodes[a]
            f.K = kf[a]
            f.J = Js[a]
            for id in clause
                i = abs(id)
                @assert id ≠ 0
                v = vnodes[i]
                push!(f.η̄list, MessU())
                push!(f.vlist, v)
            end
        end
        for (a,f) in enumerate(fnodes)
            @assert length(f.η̄list) == kf[a]
            @assert length(f.vlist) == kf[a]
        end

        σ = ones(Int, N)
        fperm = collect(1:M)
        new(N, M, fnodes, vnodes, σ, cnf, fperm)
    end
end

type ReinfParams
    reinf::Float64
    step::Float64
    wait_count::Int
    ReinfParams(reinf=0., step = 0.) = new(reinf, step, 0)
end

function initrand!(g::FactorGraphKSAT)
    for v in g.vnodes
        reset!(v)
    end

    for f in g.fnodes
        @extract f K J η̄list vlist
        for k = 1:K
            r = 0.5 * rand()
            η̄ = r / (1 - r)
            v = vlist[k]
            if J[k] == 1
                v.πp *= η̄
            else
                v.πm *= η̄
            end
            η̄list[k] = η̄
        end
    end
end

function update!(f::Fact)
    @extract f K J η̄list vlist ζ
    Δ = 1.
    ζprod = 1.
    eps = 1e-15
    nzeros = 0
    @inbounds for i = 1:K
        #v = vlist[i]
        πp = vlist[i].πp
        πm = vlist[i].πm
        if J[i] == 1
            πu = πp / η̄list[i]
            πs = πm
        else
            πu = πm / η̄list[i]
            πs = πp
        end
        z = πu / (πu + πs)
        if z > eps
            ζprod *= z
        else
            nzeros += 1
        end
        ζ[i] = z
    end
    @inbounds for i = 1:K
        if nzeros == 0
            η = ζprod / ζ[i]
        elseif nzeros == 1 && ζ[i] < eps
            η = ζprod
        else
            η = 0.
        end
        η̄new = 1 - η
        η̄old = η̄list[i]
        v = vlist[i]
        if J[i] == 1
            v.πp *= η̄new / η̄old
        else
            v.πm *= η̄new / η̄old
        end
        η̄list[i] = η̄new
        # Δ = max(Δ, abs(ηi  - old))
    end
    return Δ
end

function update!(v::Var, reinf::Float64 = 0.)

    #### update reinforcement ######

    v.πp /= v.η̄reinfp
    v.πm /= v.η̄reinfm

    if v.πp0 < v.πm0
        v.η̄reinfp = (v.πp0 / v.πm0)^reinf
        v.η̄reinfm = 1.0
    else
        v.η̄reinfm = (v.πm0 / v.πp0)^reinf
        v.η̄reinfp = 1.0
    end

    v.πp *= v.η̄reinfp
    v.πm *= v.η̄reinfm

    v.πp0 = v.πp
    v.πm0 = v.πm

    return
end

function oneBPiter!(g::FactorGraph, reinf::Float64=0.)
    @extract g fperm
    Δ = 1.

    shuffle!(fperm)
    for a in fperm
        d = update!(g.fnodes[a])
        #Δ = max(Δ, d)
    end

    for v in g.vnodes
        update!(v, reinf)
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
        v = vnodes[i]
        σ[i] = 1 - 2 * (v.πp > v.πm)
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
    return copy(getconfig!(g))
end
