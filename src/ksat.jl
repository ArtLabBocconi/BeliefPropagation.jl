using MacroUtils
include("cnf.jl")
typealias MessU Float64  # ̂ν(a→i) = P(σ_i != J_ai)
typealias MessH Float64 #  ν(i→a) = P(σ_i != J_ai)
MessU()= MessU(0.)

typealias PU Ptr{MessU}
typealias PH Ptr{MessH}
getref(v::Vector, i::Integer) = pointer(v, i)

# typealias PU Ref{MessU}
# typealias PH Ref{MessH}
# getref(v::Vector, i::Integer) = Ref(v, i)


Base.getindex(p::Ptr) = unsafe_load(p)
Base.setindex!{T}(p::Ptr{T}, x::T) = unsafe_store!(p, x)
typealias VU Vector{MessU}
typealias VH Vector{MessH}
typealias VRU Vector{PU}
typealias VRH Vector{PH}

type Fact
    πlist::Vector{MessH}
    ηlist::VRU
end
Fact() = Fact(VH(), VRU())

type Var
    ηlistp::Vector{MessU}
    ηlistm::Vector{MessU}
    πlistp::VRH
    πlistm::VRH

    #used only in BP+reinforcement
    ηreinfp::MessU
    ηreinfm::MessU
end

Var() = Var(VU(),VU(), VRH(), VRH(), 1., 1.)

abstract FactorGraph
type FactorGraphKSAT <: FactorGraph
    N::Int
    M::Int
    fnodes::Vector{Fact}
    vnodes::Vector{Var}
    cnf::CNF

    function FactorGraphKSAT(cnf::CNF)
        @extract cnf M N clauses
        println("# read CNF formula")
        println("# N=$N M=$M α=$(M/N)")
        fnodes = [Fact() for i=1:M]
        vnodes = [Var() for i=1:N]
        kf = map(length, clauses)
        kvp = zeros(Int, N)
        kvm = zeros(Int, N)
        for clause in clauses
            for id in clause
                if id > 0
                    kvm[abs(id)] += 1
                else
                    kvp[abs(id)] += 1
                end
            end
        end

        ## Reserve memory in order to avoid invalidation of Refs
        for (a,f) in enumerate(fnodes)
            sizehint!(f.πlist, kf[a])
            sizehint!(f.ηlist, kf[a])
        end
        for (i,v) in enumerate(vnodes)
            sizehint!(v.ηlistm, kvm[i])
            sizehint!(v.ηlistp, kvp[i])
            sizehint!(v.πlistm, kvm[i])
            sizehint!(v.πlistp, kvp[i])
        end

        for (a, clause) in enumerate(clauses)
            for id in clause
                i = abs(id)
                assert(id != 0)
                f = fnodes[a]
                v = vnodes[i]
                if id > 0
                    push!(v.ηlistm, MessU())
                    push!(f.ηlist, getref(v.ηlistm,length(v.ηlistm)))

                    push!(f.πlist, MessH())
                    push!(v.πlistm, getref(f.πlist,length(f.πlist)))
                else
                    push!(v.ηlistp, MessU())
                    push!(f.ηlist, getref(v.ηlistp,length(v.ηlistp)))

                    push!(f.πlist, MessH())
                    push!(v.πlistp, getref(f.πlist,length(f.πlist)))
                end
            end
        end
        new(N, M, fnodes, vnodes, cnf)
    end
end

type ReinfParams
    reinf::Float64
    step::Float64
    wait_count::Int
    ReinfParams(reinf=0., step = 0.) = new(reinf, step, 0)
end

deg(f::Fact) = length(f.ηlist)
degp(v::Var) = length(v.ηlistp)
degm(v::Var) = length(v.ηlistm)

function initrand!(g::FactorGraphKSAT)
    for f in g.fnodes
        for k=1:deg(f)
            f.πlist[k] = rand()
        end
    end
    for v in g.vnodes
        for k=1:degp(v)
            r = 0.5*rand()
            v.ηlistp[k] = 1 - (1-2r)/(1-r)
        end
        for k=1:degm(v)
            r = 0.5*rand()
            v.ηlistm[k] = 1 - (1-2r)/(1-r)
        end
        v.ηreinfm = 1
        v.ηreinfp = 1
    end
end

function update!(f::Fact)
    @extract f ηlist πlist
    Δ = 1.
    η = 1.
    eps = 1e-15
    nzeros = 0
    for i=1:deg(f)
        if πlist[i] > eps
            η *= πlist[i]
        else
            # println("here")
            nzeros+=1
        end
    end
    for i=1:deg(f)
        if nzeros == 0
            ηi = η / πlist[i]
        elseif nzeros == 1 && πlist[i] < eps
            ηi = η
        else
            ηi = 0.
        end
        # old = ηlist[i][]
        ηlist[i][] = 1 - ηi
        # Δ = max(Δ, abs(ηi  - old))
    end
    Δ
end

function update!(v::Var, reinf::Float64 = 0.)
    #TODO check del denominatore=0
    @extract v ηlistp ηlistm πlistp πlistm
    Δ = 1.
    eps = 1e-15

    ### compute total fields
    πp, πm = πpm(v)

    ### compute cavity fields
    for i=1:degp(v)
        πpi = πp / ηlistp[i]
        πlistp[i][] = πpi  / (πpi + πm)
    end

    for i=1:degm(v)
        πmi = πm / ηlistm[i]
        πlistm[i][] = πmi / (πmi + πp)
    end
    ###############

    #### update reinforcement ######
    if πp < πm
        v.ηreinfp = (πp/πm)^reinf
        v.ηreinfm = 1
    else
        v.ηreinfm = (πm/πp)^reinf
        v.ηreinfp = 1
    end
    #########################

    # #### update pseudo-reinforcement ######
    # πpR = πp / v.ηreinfp
    # πmR = πm / v.ηreinfm
    # mγ = (πpR-πmR) / (πpR+πmR) * tγ
    # pp = (1+mγ)^reinf
    # mm = (1-mγ)^reinf
    # mR = tγ * (pp-mm) / (pp+mm)
    # if πp < πm
    #     v.ηreinfp = 1 - 2mR / (mR - 1)
    #     v.ηreinfm = 1
    # else
    #     v.ηreinfm = 1 - 2mR / (1 + mR)
    #     v.ηreinfp = 1
    # end
    # #########################

    Δ
end

function oneBPiter!(g::FactorGraph, reinf::Float64=0.)
    Δ = 0.

    for a=randperm(g.M)
        d = update!(g.fnodes[a])
        Δ = max(Δ, d)
    end

    for i=randperm(g.N)
        d = update!(g.vnodes[i], reinf)
        Δ = max(Δ, d)
    end

    Δ
end

function update_reinforcement!(reinfpar::ReinfParams)
    if reinfpar.wait_count < 4
        reinfpar.wait_count += 1
    else
        reinfpar.reinf = 1 - (1-reinfpar.reinf) * (1-reinfpar.step)
    end
end

function getconfig(g::FactorGraph)
    m =  [m for m in mags(g)]
    return Int[1-2signbit(m) for m in m]
end

function converge!(g::FactorGraph; maxiters::Int = 100, ϵ::Float64=1e-5, reinfpar::ReinfParams=ReinfParams())
    for it=1:maxiters
        write("it=$it ... ")
        Δ = oneBPiter!(g, reinfpar.reinf)
        σ = getconfig(g)
        E = energy(g.cnf, σ)
        @printf("reinf=%.3f E=%d  Δ=%f \n",reinfpar.reinf, E, Δ)
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
    E = 0
    for c in cnf.clauses
        issatisfied = false
        for i in c
            if sign(i) == σ[abs(i)]
                issatisfied = true
            end
        end
        E += issatisfied ? 0 : 1
    end
    E
end

function πpm(v::Var)
    @extract v ηlistp ηlistm
    πp = 1.
    for η in ηlistp
        πp *= η
    end
    πm = 1.
    for η in ηlistm
        πm *= η
    end
    # (nzp > 0 && nzm > 0) && exit("contradiction")
    πp *= v.ηreinfp
    πm *= v.ηreinfm

    return πp, πm
end

function mag(v::Var)
    πp, πm = πpm(v)
    return (πp - πm) / (πm + πp)
end

mags(g::FactorGraph) = Float64[mag(v) for v in g.vnodes]

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
    return getconfig(g)
end
