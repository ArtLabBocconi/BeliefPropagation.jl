using ExtractMacro
using Printf
using Random

include("cnf.jl")

const MessU = Float64  # ̂ν(a→i) = P(σ_i != J_ai)
const MessH = Float64 #  ν(i→a) = P(σ_i != J_ai)
getref(v::Vector, i::Integer) = pointer(v, i)

const PU = Ptr{MessU}
const PH = Ptr{MessH}

const VU = Vector{MessU}
const VH = Vector{MessH}
const VRU = Vector{PU}
const VRH = Vector{PH}

mutable struct Fact
    πlist::Vector{MessH}
    ηlist::VRU
end
Fact() = Fact(VH(), VRU())

mutable struct Var
    pinned::Int
    ηlistp::Vector{MessU}
    ηlistm::Vector{MessU}
    πlistp::VRH
    πlistm::VRH

    #used only in BP+reinforcement
    ηreinfp::MessU
    ηreinfm::MessU
end

Var() = Var(0, VU(),VU(), VRH(), VRH(), 1., 1.)

abstract type FactorGraph end
mutable struct FactorGraphKSAT <: FactorGraph
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
                @assert(id != 0)
                f = fnodes[a]
                v = vnodes[i]
                if id > 0
                    push!(v.ηlistm, MessU(0))
                    push!(f.ηlist, getref(v.ηlistm,length(v.ηlistm)))

                    push!(f.πlist, MessH(0))
                    push!(v.πlistm, getref(f.πlist,length(f.πlist)))
                else
                    push!(v.ηlistp, MessU(0))
                    push!(f.ηlist, getref(v.ηlistp,length(v.ηlistp)))

                    push!(f.πlist, MessH(0))
                    push!(v.πlistp, getref(f.πlist,length(f.πlist)))
                end
            end
        end
        new(N, M, fnodes, vnodes, cnf)
    end
end

mutable struct ReinfParams
    r::Float64
    rstep::Float64
    γ::Float64
    γstep::Float64
    tγ::Float64
    wait_count::Int
    ReinfParams(r=0.,rstep=0.,γ=0.,γstep=0.) = new(r, rstep, γ, γstep, tanh(γ))
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
    η = 1.
    eps = 1e-15
    nzeros = 0
    for i=1:deg(f)
        if πlist[i] > eps
            η *= πlist[i]
        else
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
        ηlist[i][] = 1 - ηi
    end
end

setfree!(v::Var) = v.pinned = 0

function setpinned!(v::Var, σ::Int)
    #TODO check del denominatore=0
    @extract v  πlistp πlistm
    v.pinned = σ
    ### compute cavity fields
    for i=1:degp(v)
        πlistp[i][] = σ > 0 ? 1. : 0.
    end

    for i=1:degm(v)
        πlistm[i][] = σ > 0 ? 0. : 1.
    end
end

ispinned(v::Var) = v.pinned != 0
numpinned(g::FactorGraphKSAT) = sum(ispinned, g.vnodes)

# r = fraction of N to assign
function pin_most_biased!(g::FactorGraphKSAT, r::Float64 = 0.02)
    mlist = Vector{Tuple{Int,Float64}}()
    npin = numpinned(g)
    sizehint!(mlist, g.N - npin)
    for (i,v) in enumerate(g.vnodes)
        if !ispinned(v)
            push!(mlist, (i, mag(v)))
        end
    end

    ntopin = min(ceil(Int, r*g.N), length(mlist))
    println("# Pinning $ntopin Variables")
    sort!(mlist, lt = (x,y)->abs(x[2]) > abs(y[2]))
    for k=1:ntopin
        i, m = mlist[k]
        setpinned!(g.vnodes[i], 1-2signbit(m))
    end
end

# r = fraction of N to free
function free_most_frustated!(g::FactorGraphKSAT, r::Float64 = 0.01)
    mlist = Vector{Tuple{Int,Float64}}()
    npin = numpinned(g)
    sizehint!(mlist, npin)
    for (i,v) in enumerate(g.vnodes)
        if ispinned(v)
            σ = v.pinned
            v.pinned = 0. # == setfree!(v)
            push!(mlist, (i, σ*mag(v)))
            v.pinned = σ
        end
    end

    ntofree = min(ceil(Int, r*g.N), length(mlist))
    println("# Freeing $ntofree Variables")
    sort!(mlist, lt = (x,y)->x[2] < y[2])
    for k=1:ntofree
        i, m = mlist[k]
        setfree!(g.vnodes[i])
    end
end

function update!(v::Var, r::Float64 = 0., tγ::Float64 = 0.)
    #TODO check del denominatore=0
    ispinned(v) && return 0.
    @extract v ηlistp ηlistm πlistp πlistm
    Δ = 0.
    ### compute total fields
    πp, πm = πpm(v)

    ### compute cavity fields
    for i=1:degp(v)
        πpi = πp / ηlistp[i]
        πpi /= (πpi + πm)
        old = πlistp[i][]
        πlistp[i][] = πpi
        Δ = max(Δ, abs(πpi- old))
    end

    for i=1:degm(v)
        πmi = πm / ηlistm[i]
        πmi /= (πp + πmi)
        old = πlistm[i][]
        πlistm[i][] = πmi
        Δ = max(Δ, abs(πmi- old))
    end
    ###############

    if tγ == 0.
        #### reinforcement ######
        if πp < πm
            v.ηreinfp = (πp/πm)^r
            v.ηreinfm = 1
        else
            v.ηreinfm = (πm/πp)^r
            v.ηreinfp = 1
        end
    else
        #### pseudo-reinforcement ######
        πpR = πp / v.ηreinfp
        πmR = πm / v.ηreinfm
        mγ = (πpR-πmR) / (πpR+πmR) * tγ
        pp = (1+mγ)^r
        mm = (1-mγ)^r
        mR = tγ * (pp-mm) / (pp+mm)
        if πp < πm
            v.ηreinfp = (1 + mR) / (1 - mR)
            v.ηreinfm = 1
        else
            v.ηreinfm = (1 - mR) / (1 + mR)
            v.ηreinfp = 1
        end
    end
    Δ
end

function oneBPiter!(g::FactorGraphKSAT, r::Float64=0., tγ::Float64=0.)
    Δ = 0.

    for a=randperm(g.M)
        update!(g.fnodes[a])
    end

    for i=randperm(g.N)
        d = update!(g.vnodes[i], r, tγ)
        Δ = max(Δ, d)
    end

    Δ
end

function update_reinforcement!(reinfpar::ReinfParams)
    if reinfpar.wait_count < 10
        reinfpar.wait_count += 1
    else
        if reinfpar.γ == 0.
            reinfpar.r = 1 - (1-reinfpar.r) * (1-reinfpar.rstep)
        else
            reinfpar.r *= 1 + reinfpar.rstep
            reinfpar.γ *= 1 + reinfpar.γstep
            reinfpar.tγ = tanh(reinfpar.γ)
        end
    end
end

getσ(mags::Vector) = Int[1-2signbit(m) for m in mags]

function converge!(g::FactorGraph; maxiters::Int = 100, ϵ::Float64=1e-5
        , reinfpar::ReinfParams=ReinfParams(), alt_when_solved::Bool=false)

    for it=1:maxiters
        print("it=$it ... ")
        Δ = oneBPiter!(g, reinfpar.r, reinfpar.tγ)
        E = energy(g)
        fp = numpinned(g) / g.N
        @printf("r=%.3f γ=%.3f ρ_pin=%f\t  E=%d   \tΔ=%f \n",reinfpar.r,  reinfpar.γ, fp, E, Δ)
        # σ_noreinf = getσ(mags_noreinf(g))
        # E_noreinf = energy(g.cnf, σ_noreinf)
        # @printf("r=%.3f γ=%.3f \t  E=%d   \tE_noreinf=%d   Δ=%f \n",reinfpar.r, reinfpar.γ, E, E_noreinf, Δ)
        # println(mags(g)[1:10])
        update_reinforcement!(reinfpar)
        if alt_when_solved && E == 0
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
                break
            end
        end
        E += issatisfied ? 0 : 1
    end
    E
end

energy(g::FactorGraphKSAT) = energy(g.cnf, getσ(mags(g)))

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
    ispinned(v) && return float(v.pinned)
    πp, πm = πpm(v)
    m = (πp - πm) / (πm + πp)
    @assert isfinite(m)
    return m
end

function mag_noreinf(v::Var)
    ispinned(v) && return float(v.pinned)
    πp, πm = πpm(v)
    πp /= v.ηreinfp
    πm /= v.ηreinfm
    m = (πp - πm) / (πm + πp)
    # @assert isfinite(m)
    return m
end

mags(g::FactorGraph) = Float64[mag(v) for v in g.vnodes]
mags_noreinf(g::FactorGraphKSAT) = Float64[mag_noreinf(v) for v in g.vnodes]

function solve(cnfname::AbstractString; kw...)
    cnf = readcnf(cnfname)
    solve(cnf; kw...)
end

function solve(; N::Int=1000, α::Float64=3., k::Int = 4, seed_cnf::Int=-1, kw...)
    seed_cnf > 0 && Random.seed!(seed_cnf)
    cnf = CNF(N, k, α)
    solve(cnf; kw...)
end

function solve(cnf::CNF; maxiters::Int = 10000, ϵ::Float64 = 1e-4,
                method = :reinforcement, #[:reinforcement, :decimation]
                r::Float64 = 0., rstep::Float64= 0.001,
                γ::Float64 = 0., γstep::Float64=0.,
                alt_when_solved::Bool = true,
                seed::Int = -1)
    seed > 0 && Random.seed!(seed)
    g = FactorGraphKSAT(cnf)
    initrand!(g)
    E = -1
    if method == :reinforcement
        reinfpar = ReinfParams(r, rstep, γ, γstep)
        converge!(g, maxiters=maxiters, ϵ=ϵ, reinfpar=reinfpar, alt_when_solved=alt_when_solved)
    elseif method == :decimation
        converge!(g, maxiters=maxiters, ϵ=ϵ, alt_when_solved=alt_when_solved)
        while true
            pin_most_biased!(g, r) # r=frac fixed , γ=frac freed
            free_most_frustated!(g, γ) # r=frac fixed , γ=frac freed
            converge!(g, maxiters=maxiters, ϵ=ϵ, alt_when_solved=alt_when_solved)
            E = energy(g)
            numdec = numpinned(g)
            (E == 0 || numdec == g.N) && break
        end
    end
    E = energy(g)
    return E, getσ(mags(g))
end
