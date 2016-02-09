using MacroUtils

#TODO DA FINIRE
typealias Mess Float64
typealias PMess Ptr{Mess}
typealias VMess Vector{Mess}
typealias VPMess Vector{PMess}

getref(v::Vector, i::Integer) = pointer(v, i)
getref(v::Vector) = pointer(v, 1)

Base.Ptr{T}() = convert(T, C_NULL)
Mess() = Mess(0.)

G(x) = e^(-(x^2)/2) / √(convert(typeof(x),2) * π)
H(x) = erfc(x / √convert(typeof(x),2)) / 2
#GH(x) = ifelse(x > 30.0, x+(1-2/x^2)/x, G(x) / H(x))
function GHapp(x)
    y = 1/x
    y2 = y^2
    x + y * (1 - 2y2 * (1 - 5y2 * (1 - 7.4y2)))
end
GH(x) = x > 30.0 ? GHapp(x) : G(x) / H(x)

type Fact
    m::VMess
    m̂::VPMess
    pu::VPMess
    pd::VMess
    ξ::SubArray
end
Fact(ξ) = Fact(VMess(), VPMess(), PMess(), Mess(), ξ)

type ExactFact
    pu::VMess
    pd::VPMess
    σ::Int

    #bookkeeping
    expf::Vector{Complex128}
    expinv0::Vector{Complex128}
    expinvp::Vector{Complex128}
    expinvm::Vector{Complex128}
end
function ExactFact(σ, K)
    K2 = div(K-1, 2)
    expf = Complex128[exp(2π*im*p/K) for p=0:K-1]
    expinv0 = Complex128[(-1)^p *exp(π*im*p/K) for p=0:K-1]
    expinvp = Complex128[(
            a =(-1)^p *exp(π*im*p/K);
            b = exp(-2π*im*p/K);
            p==0 ? K2 : a*b/(1-b)*(1-b^K2))
            for p=0:K-1]
    expinvm = Complex128[(
            a =(-1)^p *exp(π*im*p/K);
            b = exp(2π*im*p/K);
            p==0 ? K2 : a*b/(1-b)*(1-b^K2))
            for p=0:K-1]

    ExactFact(VMess(), VPMess(), σ, expf, expinv0, expinvp, expinvm)
end

type Var
    m̂::VMess
    m::VPMess
    #used only in BP+reinforcement
    h::Mess
end

Var() = Var(VMess(), VPMess(), Mess())

type FactorGraph
    N::Int
    M::Int
    K::Int
    ξ::Matrix{Int}
    σ::Vector{Int}
    fnodes::Vector{Fact}
    vnodes::Vector{Var}
    exactf::Vector{ExactFact}

    function FactorGraph(ξ::Matrix{Int}, σ::Vector{Int}, K::Int)
        N, M = size(ξ)
        @assert size(ξ, 2) == length(σ)
        println("# N=$N M=$M α=$(M/N)")
        fnodes = [Fact(sub(ξ, :, a), σ[a]) for (a,k) in product(1:M,1:K)]
        vnodes = [Var() for (a,k) in product(1:M,1:K)]
        exactf = [ExactFact(σ[a], K) for a=1:M]
        ## Reserve memory in order to avoid invalidation of Refs
        for f in fnodes
            sizehint!(f.m, N)
            sizehint!(f.m̂, N)
        end
        for v in vnodes
            sizehint!(v.m, M)
            sizehint!(v.m̂, M)
        end
        for a=1:M, k=1:K
            ak = a + M*(k-1)
            f = fnodes[ak]
            exf = exactf[a]

            push!(exf.pu, Mess())
            push!(f.pu, getref(exf.pu, length(exf.pu)))

            push!(f.pd, Mess())
            push!(exf.pd, getref(f.pd, length(f.pd)))
        end

        for i=1:N, a=1:M, k=1:K
            ak = a + M*(k-1)
            ik = i + N*(k-1)
            f = fnodes[ak]
            v = vnodes[ik]

            push!(v.m̂, Mess())
            push!(f.m̂, getref(v.m̂, length(v.m̂)))

            push!(f.m, Mess())
            push!(v.m, getref(f.m, length(f.m)))
        end

        new(N, M, K, ξ, σ, fnodes, vnodes, exactf)
    end
end

type ReinfParams
    r::Float64
    r_step::Float64
    γ::Float64
    γ_step::Float64
    tγ::Float64
    wait_count::Int
    ReinfParams(r=0.,r_step=0.,γ=0.,γ_step=0.) = new(r, r_step, γ, γ_step, tanh(γ))
end

deg(f::Fact) = length(f.m)
deg(f::ExactFact) = length(f.pu)
deg(v::Var) = length(v.m)

function initrand!(g::FactorGraph)
    for f in g.fnodes
        f.m[:] = (2*rand(deg(f)) - 1)/2
        f.pd[:] = rand(1)
    end
    for v in g.vnodes
        v.m̂[:] = (2*rand(deg(v)) - 1)/2
    end
    for exf in g.exactf
        f.pu[:] = rand(g.K)
    end
end

function update!(f::Fact)
    @extract f m m̂ ξ pu pd
    M = 0.
    C = float(deg(f))
    for i=1:deg(f)
        M += ξ[i]*m[i]
        C -= m[i]^2
    end
    pu[1][] = H(-Mcav / √C)
    for i=1:deg(f)
        Mcav = M - ξ[i]*m[i]
        Ccav = sqrt(C - (1-m[i]^2))
        Hp = H(-Mcav / Ccav); Hm = 1-Hp
        Gp = G(-Mcav / Ccav); Gm = Gp
        m̂[i][] = ξ[i] / Ccav * (pd[1]*Gp - (1-pd[1])*Gm) / (pd[1]*Hp + (1-pd[1])*Hm)
    end
end


function update!(f::ExactFact)
    @extract f σ pu pd expf expinv0 expinvm expinvp
    K = deg(f)
    X = ones(Complex128, K)
    for p=1:K
        for k=1:K
            X[p] *= (1-pu[k]) + pu[k]*expf[p]
        end
    end
    for k=1:K
        s0 = Complex128(0.)
        sp = Complex128(0.)
        sm = Complex128(0.)
        for p=1:K
            xp = X[p] / ((1-pu[k]) + pu[k]*expf[p])
            s0 += expinv0[p] * xp
            sp += expinvp[p] * xp
            sm += expinvm[p] * xp
        end
        sr = σ > 0 ? real(s0 /(s0+2sp)) : -real(s0 /(s0+2sm))
        pd[k][] = 0.5*(1+sr)
        @assert isfinite(pd[k][])
    end
end

function update!(v::Var, r::Float64 = 0.)
    @extract v m m̂
    Δ = 0.

    v.h = sum(m̂) + r*v.h
    ### compute cavity fields
    for a=1:deg(v)
        newm = tanh(v.h - m̂[a])
        oldm = m[a][]
        m[a][] = newm
        Δ = max(Δ, abs(newm - oldm))
    end

    Δ
end

function oneBPiter!(g::FactorGraph, r::Float64=0.)
    Δ = 0.

    for f in fnodes
        update!(f)
    end

    for v in vnodes
        d = update!(v, r)
        Δ = max(Δ, d)
    end

    Δ
end

function update_reinforcement!(reinfpar::ReinfParams)
    if reinfpar.wait_count < 10
        reinfpar.wait_count += 1
    else
        if reinfpar.γ == 0.
            reinfpar.r = 1 - (1-reinfpar.r) * (1-reinfpar.r_step)
        else
            reinfpar.r *= 1 + reinfpar.r_step
            reinfpar.γ *= 1 + reinfpar.γ_step
            reinfpar.tγ = tanh(reinfpar.γ)
        end
    end
end

getW(mags::Vector) = Int[1-2signbit(m) for m in mags]

function converge!(g::FactorGraph; maxiters::Int = 10000, ϵ::Float64=1e-5, alt_when_solved::Bool=false
                                 , reinfpar::ReinfParams=ReinfParams())

    for it=1:maxiters
        write("it=$it ... ")
        Δ = oneBPiter!(g, reinfpar.r)
        E = energy(g)
        @printf("r=%.3f γ=%.3f  E=%d   \tΔ=%f \n", reinfpar.r, reinfpar.γ, E, Δ)
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

function energy(g::FactorGraph, W::Vector{Int})
    E = 0
    for a=1:M
        r=[(k-1)*M + a for k=1:K]
        σks = Int[ifelse(dot(ξ[:,a], W[r]) > 0, 1, -1) for k=1:K]
        E += g.σ[a] * sum(σks) > 0 ? 0 : 1
    end
    E
end

energy(g::FactorGraph) = energy(g, getW(mags(g)))

function mag(v::Var)
    m = tanh(v.h)
    @assert isfinite(m)
    return m
end
#
# function mag_noreinf(v::Var)
#     ispinned(v) && return float(v.pinned)
#     πp, πm = πpm(v)
#     πp /= v.ηreinfp
#     πm /= v.ηreinfm
#     m = (πp - πm) / (πm + πp)
#     # @assert isfinite(m)
#     return m
# end

mags(g::FactorGraph) = Float64[mag(v) for v in g.vnodes]
# mags_noreinf(g::FactorGraph) = Float64[mag_noreinf(v) for v in g.vnodes]


function solve(; N::Int=1000, α::Float64=0.6, seed_ξ::Int=-1, kw...)
    if seed_ξ > 0
        srand(seed_ξ)
    end
    M = round(Int, α * N)
    ξ = rand([-1,1], N, M)
    σ = ones(Int, M)
    solve(ξ, σ; kw...)
end

function solve(ξ::Matrix{Int}, σ::Vector{Int}; maxiters::Int = 10000, ϵ::Float64 = 1e-4,
                method = :reinforcement, #[:reinforcement, :decimation]
                K::Int=3,
                r::Float64 = 0., r_step::Float64= 0.001,
                γ::Float64 = 0., γ_step::Float64=0.,
                alt_when_solved::Bool = true,
                seed::Int = -1)
    @assert K % 2 == 1
    seed > 0 && srand(seed)
    g = FactorGraph(ξ, σ, K)
    initrand!(g)

    # if method == :reinforcement
    reinfpar = ReinfParams(r, r_step, γ, γ_step)
    converge!(g, maxiters=maxiters, ϵ=ϵ, reinfpar=reinfpar
            , alt_when_solved=alt_when_solved)
    return getW(mags(g))
end
