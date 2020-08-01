using ExtractMacro
using PyPlot
include("../utils/integrals.jl")
#TODO DA FINIRE
typealias Mess Float64
typealias PMess Ptr{Mess}
typealias VMess Vector{Mess}
typealias VPMess Vector{PMess}

getref(v::Vector, i::Integer) = pointer(v, i)
getref(v::Vector) = pointer(v, 1)

Base.getindex(p::Ptr) = unsafe_load(p)
Base.setindex!{T}(p::Ptr{T}, x::T) = unsafe_store!(p, x)
Base.show(io::IO, p::Ptr) = show(io, p[])
Base.show(p::Ptr) = show(p[])

Base.Ptr{T}() = convert(T, C_NULL)
Mess() = Mess(0.)

G(x) = exp(-(x^2)/2) / √(convert(typeof(x),2) * π)
H(x) = erfc(x / √convert(typeof(x),2)) / 2
E(x) = erf(x / √convert(typeof(x),2))
#GH(x) = ifelse(x > 30.0, x+(1-2/x^2)/x, G(x) / H(x))
function GHapp(x)
    y = 1/x
    y2 = y^2
    x + y * (1 - 2y2 * (1 - 5y2 * (1 - 7.4y2)))
end
GH(x) = x > 30.0 ? GHapp(x) : G(x) / H(x)

mutable struct Fact
    ## INCOMING
    m::VMess # from W ↑
    ρ::VMess # from W ↑
    mt::VMess # from TopFact ↓

    ## OUTGOING
    mh::VPMess # to W ↓
    ρh::VPMess # to W ↓
    mhb::VPMess  # to TopFact ↑
    ρhb::VPMess  # to TopFact ↑

    ξ::Vector{Float64}
end
Fact(ξ) = Fact(VMess(), VMess(),VMess(),VPMess(), VPMess(),VPMess(), VPMess(), ξ)

mutable struct TopFact
    ## INCOMING
    mb::VMess  # from Fact ↑
    ρb::VMess  # from Fact ↑

    ## OUTGOING
    mht::VPMess  # to Fact ↓

    σ::Int
end
TopFact(σ) = TopFact(VMess(),VMess(),VPMess(), σ)

mutable struct Var
    ## INCOMING
    mh::VMess # from Fact ↓
    ρh::VMess # from Fact ↓

    ## OUTGOING
    m::VPMess # to Fact  ↑
    ρ::VPMess # to Fact ↑

    #used only in BP+reinforcement
    h1::Mess
    h2::Mess
    λ::Float64
end
Var(λ=1.) = Var(VMess(), VMess(),
                    VPMess(), VPMess(), 0.,0., λ)

mutable struct VarY
    ## INCOMING
    mht::VMess # from Fact ↓
    mhb::VMess # from Fact ↑
    ρhb::VMess # from Fact ↑

    ## OUTGOING
    mt::VPMess # to Fact ↓
    mb::VPMess # to Fact ↑
    ρb::VPMess # to Fact ↑

    #used only in BP+reinforcement
    h1::Mess
    h2::Mess
    λ::Float64
end

VarY(λ=1.) = VarY(VMess(), VMess(), VMess(),
                    VPMess(), VPMess(), VPMess(), 0.,0., λ)

mutable struct FactorGraph
    N::Int
    M::Int
    K::Int
    ξ::Matrix{Int}
    σ::Vector{Int}
    fnodes::Vector{Vector{Fact}} # fnodes[μ][i]= ψ_μi
    ynodes::Vector{Vector{VarY}} # vnodes[i][j] = y_μi
    vnodes::Vector{Vector{Var}} # vnodes[i][j] = W_ij
    topfnodes::Vector{TopFact}  # topfnodes[μ]= ψ_μ

    activation::Symbol
    weight::Symbol


    function FactorGraph(ξ::Matrix{Int}, σ::Vector{Int}, K::Int
            ; λ=1., activation=:erf,weight=:continuous)
        N, M = size(ξ)
        @assert size(ξ, 2) == length(σ)
        println("# N=$N M=$M α=$(M/N)")
        fnodes = [[Fact(ξ[:, a]) for i=1:K] for a=1:M]
        ynodes = [[VarY(0.) for i=1:K] for a=1:M]
        vnodes = [[Var(λ) for  j=1:N] for i=1:K]
        topfnodes = [TopFact(σ[a]) for a=1:M]

        ## Reserve memory in order to avoid invalidation of Refs
        ## TODO put in a separate function
        for a=1:M
            for k=1:K
                f = fnodes[a][k]
                ## INCOMING
                sizehint!(f.m, N)
                sizehint!(f.ρ, N)
                sizehint!(f.mt, 1)
                ## OUTGOING
                sizehint!(f.mh, N)
                sizehint!(f.ρh, N)
                sizehint!(f.mhb, 1)
                sizehint!(f.ρhb, 1)

                v = ynodes[a][k]
                ## INCOMING
                sizehint!(v.mhb, 1)
                sizehint!(v.ρhb, 1)
                sizehint!(v.mhb, 1)
                ## OUTGOING
                sizehint!(v.mb, 1)
                sizehint!(v.ρb, 1)
                sizehint!(v.mt, 1)
            end
        end
        for i=1:K
            for v in vnodes[i]
                ## INCOMING
                sizehint!(v.mh, M)
                sizehint!(v.ρh, M)

                ## OUTGOING
                sizehint!(v.m, M)
                sizehint!(v.ρ, M)
            end
        end
        for f in topfnodes
            ## INCOMING
            sizehint!(f.mb, K)
            sizehint!(f.ρb, K)
            ## OUTGOING
            sizehint!(f.mht, K)
        end
        ####################################


        #### Make all connections ##########
        # TopFact <-> VarY
        for a=1:M, k=1:K
            f = ynodes[a][k]
            topf = topfnodes[a]

            push!(topf.mb, Mess())
            push!(f.mb, getref(topf.mb, length(topf.mb)))
            push!(topf.ρb, Mess())
            push!(f.ρb, getref(topf.ρb, length(topf.ρb)))

            push!(f.mht, Mess())
            push!(topf.mht, getref(f.mht, length(f.mht)))
        end
        # VarY <-> Fact
        for a=1:M, k=1:K
            f = fnodes[a][k]
            v = ynodes[a][k]

            push!(v.mhb, Mess())
            push!(f.mhb, getref(v.mhb, length(v.mhb)))
            push!(v.ρhb, Mess())
            push!(f.ρhb, getref(v.ρhb, length(v.ρhb)))

            push!(f.mt, Mess())
            push!(v.mt, getref(f.mt, length(f.mt)))
        end

        # Fact <-> Var
        for i=1:N, a=1:M, k=1:K
            f = fnodes[a][k]
            v = vnodes[k][i]

            push!(v.mh, Mess())
            push!(f.mh, getref(v.mh, length(v.mh)))
            push!(v.ρh, Mess())
            push!(f.ρh, getref(v.ρh, length(v.ρh)))

            push!(f.m, Mess())
            push!(v.m, getref(f.m, length(f.m)))
            push!(f.ρ, Mess())
            push!(v.ρ, getref(f.ρ, length(f.ρ)))
        end
        ################################################

        new(N, M, K, ξ, σ, fnodes, ynodes, vnodes, topfnodes, activation, weight)
    end
end

mutable struct ReinfParams
    r::Float64
    r_step::Float64
    γ::Float64
    γ_step::Float64
    tγ::Float64
    wait_count::Int
    ReinfParams(r=0.,r_step=0.,γ=0.,γ_step=0.) = new(r, r_step, γ, γ_step, tanh(γ), 0)
end

deg(f::Fact) = length(f.m)
deg(f::TopFact) = length(f.mb)
deg(v::Var) = length(v.m)

function initrand!(g::FactorGraph)
    @extract g fnodes vnodes ynodes topfnodes N K M
    ϵ =1e-3
    for a=1:M,k=1:K
        f=fnodes[a][k]
        f.m .= (2*rand(deg(f)) .- 1)
        # f.m .= ϵ
        # f.ρ .= f.m.^2 .+ 1e-3
        f.ρ  .= 1.
        # f.mt .= ϵ*rand(1)
        f.mt .= 1e-5

        v = ynodes[a][k]
        v.mhb .= 2rand() .- 1
        v.ρhb .= 1
        v.mht .= 2rand() .- 1
        v.h1 = 2rand() - 1
        v.h2 = 2rand() - 1
    end

    for k=1:K
        for v in vnodes[k]
            v.mh .= ϵ*(2*rand(deg(v)) .- 1)
            # v.mh .= ϵ
            v.ρh  .= 1e-3
            v.h1 = 2rand() - 1
            v.h2 = 2rand() - 1
        end
    end

    for f in topfnodes
        # f.mb .= ϵ*(2*rand(deg(f)) .- 1)/2
        f.mb .= 1-ϵ
        # f.ρb .= f.mb.^2 .+ 1e-3
        f.ρb .= 1.
    end
end

function update!(f::Fact, activation=:sgn)
    @extract f m mt mh mhb ρ ρh ρhb ξ
    M = 0.
    C = 0.
    for i=1:deg(f)
        M += ξ[i]*m[i]
        C += ξ[i]^2*(ρ[i]- m[i]^2)
    end
    @assert C > 0
    if activation == :sgn
        sqc = √(C)
        mhb[1][] = E(M/ sqc)
        ρhb[1][] = 1.
        @assert  isfinite(mt[1])
        p=(1+mt[1])/2
        q=(1-mt[1])/2
        # p=1.; q=0.
        for i=1:deg(f)
            Mcav = M - ξ[i]*m[i]
            x = Mcav /sqc
            Hp = H(-x); Hm = 1-Hp
            Gp = G(-x); Gm = Gp
            gh = GH(-x)
            # @assert isfinite(Gp)
            # @assert isfinite(Hp)
            if (p*Hp + q*Hm) <= 0
                println(Hp," ",x, " ",mt[1])
                println(m)
            end
            @assert (p*Hp + q*Hm) > 0
            mh[i][] = ξ[i] / sqc * (p*Gp - q*Gm) / (p*Hp + q*Hm)
            # mh[i][] = ξ[i] / sqc *gh
            @assert isfinite(mh[i][])
            #TODO nel caso di W continuo serve ρh[i][] = 0
        end
    else
        sqc1 = √(C+1)
        sqc = √(C)
        mhb[1][] = E(M/ sqc1)
        ρhb[1][] = ∫D(z->E(M +z*sqc)^2)
        for i=1:deg(f)
            Mcav = M - ξ[i]*m[i]
            g = G(Mcav/sqc1)
            mh[i][] = ξ[i] * mt[1] / sqc1 * 2*g
            ρh[i][] = ξ[i]^2 * mt[1] * Mcav /sqc1/ (C+1) * 2*g
        end
    end
end

function update!(f::TopFact)
    @extract f σ mht mb ρb
    M = 0.
    C = 0.
    for i=1:deg(f)
        M += mb[i]
        @assert abs(mb[i]) <= 1
        C += ρb[i]- mb[i]^2
    end
    @assert C>0
    sqc = √(C)
    # println(C, " " ,  M, " ",GH(-σ*M/sqc))
    for i=1:deg(f)
        Mcav = M - mb[i]
        # gh = GH(-σ*Mcav/sqc)
        # newm = σ/sqc * gh
        hp = H(-σ*(Mcav+1)/sqc)
        hm = H(-σ*(Mcav-1)/sqc)
        newm = (hp - hm) / (hp+hm)
        if !isfinite(newm)
            newm = float(1-2signbit(newm))
        end
        if abs(newm) > 1
            println("mt[i][]=", newm)
            newm = newm > 1 ? 0.1 : -.1
        end
        mht[i][] = newm
    end
end

function update!(v::Var, r::Float64 = 0., weight=:binary)
    @extract v m ρ mh ρh λ
    Δ = 0.
    v.h1 = sum(mh) + r*v.h1
    @assert isfinite(v.h1)
    if weight == :binary
        for a=1:deg(v)
            newm = tanh(v.h1 - mh[a])
            oldm = m[a][]
            m[a][] = newm
            ρ[a][] = 1.
            @assert isfinite(newm)
            Δ = max(Δ, abs(newm - oldm))
        end
    else
        v.h2 = λ + sum(ρh) + r*v.h2
        @assert v.h2 > 0
        ### compute cavity fields
        for a=1:deg(v)
            newm = (v.h1 - mh[a]) / v.h2
            oldm = m[a][]
            m[a][] = newm
            ρ[a][] = 1/v.h2 + newm^2
            Δ = max(Δ, abs(newm - oldm))
        end
    end

    Δ
end


function update!(v::VarY, r::Float64 = 0., weight=:binary)
    @extract v mt mht  mb ρb mhb ρhb λ
    mb[1][] = mhb[1]
    ρb[1][] = ρhb[1]
    mt[1][] = mht[1]
end

function oneBPiter!(g::FactorGraph, r::Float64=0.)
    @extract g fnodes vnodes ynodes topfnodes N K M
    Δ = 0.
    for a=1:M
        for k=1:K
            update!(fnodes[a][k], g.activation)
        end
    end
    for a=1:M
        for k=1:K
            update!(ynodes[a][k], r, g.weight)
        end
    end
    for f in topfnodes
        update!(f)
    end
    for a=1:M
        for k=1:K
            update!(ynodes[a][k], r, g.weight)
        end

    end
    for k=1:K
        for v in vnodes[k]
            d = update!(v, r, g.weight)
            Δ = max(Δ, d)
        end
    end

    Δ
end

function update_reinforcement!(reinfpar::ReinfParams)
    if reinfpar.wait_count < 10
        reinfpar.wait_count += 1
    elseif reinfpar.r_step > 0
        reinfpar.r = 1 - (1-reinfpar.r) * (1-reinfpar.r_step)
    else
        reinfpar.r *= 1+reinfpar.r_step
    end

end

function mag(v::Var, weight)
    m= weight==:binary ? tanh(v.h1) : v.h1 / v.h2
    @assert isfinite(m)
    return m
end
mags(g::FactorGraph) = [Float64[mag(v, g.weight) for v in W] for W in g.vnodes]

getWClipped(g::FactorGraph) = [Int[1-2signbit(m) for m in magk] for magk in mags(g)]
getW(g::FactorGraph) = mags(g)

h_top(g::FactorGraph) = [sum(f.mb) for f in g.topfnodes]
function print_overlaps{T}(g::FactorGraph, W::Vector{Vector{T}}; meanvar = true)
    K = length(W)
    N = length(W[1])
    q = reshape([dot(W[i], W[j]) / N for i=1:K,j=1:K],K^2)
    h = h_top(g)
    clf()
    subplot(211)
    plt[:hist](q)
    subplot(212)
    plt[:hist](h)

    # if meanvar
    #     println("overalaps mean,std = ",mean(q), ",", std(q))
    # else
    #     println(q)
    # end
end


function converge!(g::FactorGraph; maxiters::Int = 10000, ϵ::Float64=1e-5
                                 , altsolv::Bool=false, altconv::Bool=false
                                 , reinfpar::ReinfParams=ReinfParams())

    print_overlaps(g, getWClipped(g))
    for it=1:maxiters
        print("it=$it ... ")
        Δ = oneBPiter!(g, reinfpar.r)

        Ene = energy(g)
        EneClipp = energyClipped(g)

        @printf("r=%.3f γ=%.3f  E=%d EClipp=%d   \tΔ=%f \n",
                reinfpar.r, reinfpar.γ, Ene, EneClipp, Δ)
        print_overlaps(g, getWClipped(g))
        update_reinforcement!(reinfpar)

        if altsolv &&
            ((g.weight == :continuous && Ene == 0)
            || (g.weight == :binary && EneClipp == 0))
            println("Found Solution!")
            break
        end
        if altconv && Δ < ϵ
            println("Converged!")
            break
        end
    end
end

function energy{T}(g::FactorGraph, W::Vector{Vector{T}})
    @extract g K M  ξ σ
    ene = 0
    for a=1:M
        y  = Float64[E(dot(ξ[:,a], W[k])) for k=1:K]
        ene += g.σ[a] * sum(y) > 0 ? 0 : 1
    end
    ene
end

energy(g::FactorGraph) = energy(g, getW(g))
energyClipped(g::FactorGraph) = energy(g, getWClipped(g))


function solve(; N::Int=1000, α::Float64=0.6, K::Int=3, seed_ξ::Int=-1, kw...)
    seed_ξ > 0 && Random.seed!(seed_ξ)

    M = round(Int, α * K*N)
    M = round(Int, α *N)
    ξ = rand([-1,1], N, M)
    σ = ones(Int, M)
    solve(ξ, σ; K=K, kw...)
end

function solve(ξ::Matrix, σ::Vector{Int};
                maxiters::Int = 10000, ϵ::Float64 = 1e-4,
                K::Int=3, λ=1.,
                r::Float64 = 0., r_step::Float64= 0.001,
                γ::Float64 = 0., γ_step::Float64=0.,
                altsolv::Bool = true, altconv=false,
                seed::Int = -1,
                weight=:continuous, #:continuous, :binary
                activation=:erf #:erf, :sign
                )
    seed > 0 && Random.seed!(seed)

    g = FactorGraph(ξ, σ, K, λ=λ, activation=activation, weight=weight)
    initrand!(g)
    # if method == :reinforcement
    reinfpar = ReinfParams(r, r_step, γ, γ_step)
    converge!(g, maxiters=maxiters, ϵ=ϵ, reinfpar=reinfpar
            , altsolv=altsolv, altconv=altconv)
    return g, getW(g)
end
