module Pseudo

include("../../utils/MagnetizationsT.jl")
using .MagnetizationsT
include("../../utils/functions.jl")

using ExtractMacro
import Base: *,/
typealias Mess Mag64
typealias P Ptr{Mess}
typealias VMess Vector{Mess}
typealias VPMess Vector{P}

Base.getindex(p::Ptr) = unsafe_load(p)
Base.setindex!{T}(p::Ptr{T}, x::T) = unsafe_store!(p, x)
Base.setindex!{T}(p::Ptr{T}, x) = unsafe_store!(p, convert(T,x))
Base.show(io::IO, p::Ptr) = show(io, p[])
Base.show(p::Ptr) = show(p[])

getref(v::Vector, i::Integer) = pointer(v, i)
Mess() = Mess(0.)


# G(x) = e^(-(x^2)/2) / √(convert(typeof(x),2) * π)
# H(x) = erfc(x / √convert(typeof(x),2)) / 2
# #GH(x) = ifelse(x > 30.0, x+(1-2/x^2)/x, G(x) / H(x))
# function GHapp(x)
#     # print("ghapp")
#     y = 1/x
#     y2 = y^2
#     x + y * (1 - 2y2 * (1 - 5y2 * (1 - 7.4y2)))
# end
# GH(x) = x > 30.0 ? GHapp(x) : G(x) / H(x)
#
# G(x, β) = (1 - exp(-β)) * G(x)
# H(x,β) = (eb=exp(-β); eb + (1-eb)*H(x))
# GH(x, β) = β == Inf ? GH(x) : GHapp(x, β) #x > 30.0 ? GHapp(x, β) : G(x, β) / H(x, β)
# # function GHapp(x, β)
# #     # print("ghapp")
# #     # NOTE: not a very good approximation when x is large and β is not
# #     y = 1/x
# #     y2 = y^2
# #     a = e^(-β + (x^2)/2) / ((1 - e^(-β)) * √(2π))
# #     return x / (x * a + 1 - y2 * (1 - 3y2 * (1 - 5y2)))
# # end
# GHapp(x, β) = exp(log(G(x, β)) - log(H(x, β)))

type Fact
    m::VMess
    mt::VMess
    mh::VPMess
    mht::VPMess
    ξ::Vector{Float64}
    σ::Int
    mhtot::Mess
    mhttot::Mess
end

Fact(ξ, σ) = Fact(VMess(),VMess(), VPMess(), VPMess(), ξ, σ, 0.,0.)

type Var
    mh::VMess
    mht::VMess
    m::VPMess
    mt::VPMess

    mtot::Mess
    mttot::Mess

    uToC::Mess # u to Center
    uToP::Mess # u to Periphery
end

Var() = Var(VMess(), VMess(), VPMess(), VPMess(), zeros(4)...)

type ReinfParams
    y::Float64
    ystep::Float64
    γ::Float64
    γstep::Float64
    ReinfParams(y,ystep,γ,γstep) = new(y, ystep, γ, γstep)
end
ReinfParams() = ReinfParams(zeros(4)...)

type FactorGraph
    N::Int
    M::Int
    β::Float64
    ξ::Matrix{Float64}
    σ::Vector{Int}
    fnodes::Vector{Fact}
    vnodes::Vector{Var}
    reinf::ReinfParams

    function FactorGraph(ξ::Matrix, σ::Vector{Int}, λ::Float64=1.;β=30.)
        N = size(ξ, 1)
        M = length(σ)
        @assert size(ξ, 2) == M
        println("# N=$N M=$M α=$(M/N)")
        fnodes = [Fact(ξ[:,a], σ[a]) for a=1:M]
        vnodes = [Var() for i=1:N]

        ## Reserve memory in order to avoid invalidation of Refs
        for (a,f) in enumerate(fnodes)
            sizehint!(f.m, N)
            sizehint!(f.mt, N)
            sizehint!(f.mh, N)
            sizehint!(f.mht, N)
        end
        for (i,v) in enumerate(vnodes)
            sizehint!(v.m, M)
            sizehint!(v.mt, M)
            sizehint!(v.mh, M)
            sizehint!(v.mht, M)
        end

        for i=1:N, a=1:M
            f = fnodes[a]
            v = vnodes[i]

            push!(v.mh, Mess())
            push!(v.mht, Mess())
            push!(f.mh, getref(v.mh, length(v.mh)))
            push!(f.mht, getref(v.mht, length(v.mht)))

            push!(f.m, Mess())
            push!(f.mt, Mess())
            push!(v.m, getref(f.m, length(f.m)))
            push!(v.mt, getref(f.mt, length(f.mt)))
        end

        new(N, M, β, ξ, σ, fnodes, vnodes, ReinfParams())
    end
end

deg(f::Fact) = length(f.m)
deg(v::Var) = length(v.m)

function initrand!(g::FactorGraph)
    ϵ = 1e-1
    for f in g.fnodes
        f.m[:] = randn()*ϵ
        f.mt[:] = randn()*ϵ
    end
    for v in g.vnodes
        v.mh[:] = randn()*ϵ
        v.mht[:] = randn()*ϵ
    end
end


let mdict = Dict{Int,Vector{Float64}}(), mtdict = Dict{Int,Vector{Float64}}()
    global update!
    function update!(f::Fact, β)
        @extract f mh mht σ ξ
        uσ = mtanh(σ*Inf)

        m = Base.@get!(mdict, deg(f), Array(Float64, deg(f)))
        mt = Base.@get!(mtdict, deg(f), Array(Float64, deg(f)))
        @inbounds for i=1:deg(f)
            m[i] = f.m[i] # trasformo i Mag64 in float per non fare tanh ogni volta
            mt[i] = f.mt[i] # trasformo i Mag64 in float per non fare tanh ogni volta
        end

        M = Mt = C = Ct = 0.
        @inbounds for i=1:deg(f)
            M += ξ[i]*m[i]
            C += ξ[i]^2*(1-m[i]^2)
            Mt += ξ[i]*mt[i]
            Ct += ξ[i]^2*(1-mt[i]^2)
        end

        sqC = √(2C)
        mp = M / sqC
        f.mhtot = erfmix(uσ, mp, mp)

        sqC = √(2Ct)
        mp = Mt / sqC
        f.mhttot = erfmix(uσ, mp, mp)

        dump = 0.5
        @inbounds for i=1:deg(f)
            Mcav = M - ξ[i]*m[i]
            Ccav = C - ξ[i]^2*(1-m[i]^2)

            #sqC = √(2Ccav)
            # mp = (Mcav + ξ[i]) / sqC
            # mm = (Mcav - ξ[i]) / sqC
            # newm = erfmix(uσ, mp, mm)
            sqC = √(Ccav)
            x = Mcav / sqC
            newm = mtanh(ξ[i] / sqC * GH(atanh(uσ), -x))

            mh[i][] = damp(newm, mh[i][], dump)

            Mcav = Mt - ξ[i]*mt[i]
            Ccav = Ct - ξ[i]^2*(1-mt[i]^2)
            # sqC = √(2Ccav)
            # mp = (Mcav + ξ[i]) / sqC
            # mm = (Mcav - ξ[i]) / sqC
            # newm = erfmix(uσ, mp, mm)
            #
            sqC = √(Ccav)
            x = Mcav / sqC
            newm = mtanh(ξ[i] / sqC * GH(atanh(uσ), -x))

            mht[i][] = damp(newm, mht[i][], dump)
        end
    end

end #let

function update!(v::Var, y::Float64, γ::Float64)
    @extract v m mh mt mht
    Δ = 0.
    # h = reduce(⊗, mh)
    # ht = reduce(⊗, mht)
    h = Mess(0.); ht = Mess(0.);
    @inbounds for i=1:deg(v)
        h = h ⊗ mh[i]
        ht = ht ⊗ mht[i]
    end
    pol = mtanh(γ)
    v.uToC = pol * h
    v.uToP = pol * (ht ⊗ v.uToC↑(y-1))

    h = h ⊗ v.uToP
    ht = ht ⊗ v.uToC↑y

    Δ = max(Δ, abs(h - v.mtot), abs(ht - v.mttot))
    v.mtot = h
    v.mttot = ht
    ### compute cavity fields
    # update m
    for a=1:deg(v)
        m[a][] = h ⊘ mh[a]
    end
    # update mt
    for a=1:deg(v)
        mt[a][] = ht ⊘ mht[a]
    end

    Δ
end

function oneBPiter!(g::FactorGraph, reinf::ReinfParams)
    Δ = 0.

    for a=randperm(g.M)
        update!(g.fnodes[a], g.β)
    end

    for i=randperm(g.N)
        d = update!(g.vnodes[i], reinf.y, reinf.γ)
        Δ = max(Δ, d)
    end

    Δ
end


function update_reinforcement!(reinf::ReinfParams)
    reinf.y *= 1 + reinf.ystep
    reinf.γ += reinf.γstep
    updated = !(reinf.ystep == 0. && reinf.γstep == 0)
    return updated
end

getW(mags::Vector) = Int[1-2signbit(m) for m in mags]

function converge!(g::FactorGraph; maxiters::Int=10000, ϵ::Float64=1e-5
                                , altsolv=false, altconv=true
                                , reinf::ReinfParams=ReinfParams())

    for it=1:maxiters
        print("it=$it ... ")
        Δ = oneBPiter!(g, reinf)
        E = energy(g)
        Et = energyt(g)
        @printf("y=%.3f γ=%.3f  E(W̃)=%d E(W)=%d   \tΔ=%f \n", reinf.y, reinf.γ, Et, E, Δ)
        if altsolv && (E == 0 || Et == 0)
            println("Found Solution!")
            break
        end
        if altconv && Δ < ϵ
            println("Converged!")
            break
        end
    end
end

function energy(g::FactorGraph, W::Vector)
    E = 0
    for f in g.fnodes
        E += f.σ * dot(f.ξ, W) > 0 ? 0 : 1
    end
    E
end

energy(g::FactorGraph) = energy(g, getW(mags(g)))
energyt(g::FactorGraph) = energy(g, getW(magst(g)))

mag(v::Var) = Float64(v.mtot)
magt(v::Var) = Float64(v.mttot)
magh(f::Fact) = Float64(f.mhtot)
maght(f::Fact) = Float64(f.mhttot)

mags(g::FactorGraph) = Float64[mag(v) for v in g.vnodes]
magst(g::FactorGraph) = Float64[magt(v) for v in g.vnodes]
magsh(g::FactorGraph) = Float64[magh(f) for f in g.fnodes]
magsht(g::FactorGraph) = Float64[maght(f) for f in g.fnodes]

type ThermFunc
    ϕ::Float64 # ϕ = Σext + y*Σint  (a T=0)
    Σext::Float64
    Σint::Float64
    E::Float64
    Ẽ::Float64
end
ThermFunc()=ThermFunc(zeros(5)...)

function *(tf::ThermFunc, x::Number)
    newtf = deepcopy(tf)
    for f in fieldnames(tf)
        newtf.(f) *= x
    end
    newtf
end
/(tf::ThermFunc, x::Number) = tf*(1/x)

function free_entropy(f::Fact, y)
    @extract f: mh mht σ ξ
    uσ = mtanh(σ*Inf)
    m = map(Float64, f.m)
    mt = map(Float64, f.mt) # trasformo i Mag64 in float per non fare tanh ogni volta

    ϕext = ϕint = 0.
    M = Mt = C = Ct = 0.
    @inbounds for i=1:deg(f)
        M += ξ[i]*m[i]
        C += ξ[i]^2*(1-m[i]^2)
        Mt += ξ[i]*mt[i]
        Ct += ξ[i]^2*(1-mt[i]^2)
    end

    b = merf(M / √(2C))
    bt = merf(Mt / √(2Ct))
    ϕint += log1pxy(uσ, b)
    ϕext += log1pxy(uσ, bt)

    for i = 1:deg(f)
        ϕint -= log1pxy(f.m[i], mh[i][])
        ϕext -= log1pxy(f.mt[i], mht[i][])
    end
    ϕ = ϕext + y*ϕint
    Σint = ϕint
    return ϕ, Σint
end

function free_entropy(v::Var, y, γ)
    @extract v: mh mht m mt mtot mttot uToC uToP
    ϕext = ϕint = Σint = 0.
    pol = mtanh(γ)

    ϕint += logZ(uToP, mh)

    ϕext += logZ(uToC, mht, y)
    Σint += atanh(uToC)*v.mttot - log2cosh(atanh(uToC))

    ϕint += -log1pxy(pol, -pol) / 2 + log(2)/2
    mcav = mtot ⊘ uToP
    ϕint += -log1pxy(mcav, uToP)

    ϕ = ϕext + y*ϕint
    Σint += ϕint
    return ϕ, Σint
end

function ThermFunc(g::FactorGraph)
    @extract g: fnodes vnodes M N reinf
    @extract reinf: γ y
    ϕ = Σint = 0.
    for f in fnodes
        phi, sig = free_entropy(f, y)
        ϕ += phi
        Σint += sig
    end

    for v in vnodes
        phi, sig = free_entropy(v, y, γ)
        ϕ += phi
        Σint += sig
    end
    ϕ /= N
    Σint /= N
    op = OrderParams(g)
    Σint -= γ*op.s
    ϕ -= y*γ*op.s
    Σext = ϕ - y*Σint

    return ThermFunc(ϕ, Σext, Σint, 0., 0.)
end

type OrderParams
    m::Float64
    mt::Float64
    q0::Float64
    q1::Float64
    qt::Float64
    s::Float64
    st::Float64
    m̂::Float64
    m̂t::Float64
    q̂0::Float64
    q̂1::Float64
    q̂t::Float64
    ŝ::Float64
    ŝt::Float64
end
OrderParams()=OrderParams(zeros(14)...)

function OrderParams(g::FactorGraph)
    y, γ = g.reinf.y, g.reinf.γ
    pol = mtanh(γ)
    m = mean(mags(g))
    mt = mean(magst(g))
    mh = sum(magsh(g))
    mht = sum(magsht(g))
    st = mean(mags(g) .* magst(g))
    q0 = mean(mags(g).^2)
    qt = mean(magst(g).^2)
    q1=0.
    s = 0.
    for v in g.vnodes
        m1 = Float64(v.mtot ⊘ v.uToP)
        m2 = Float64(v.mtot ⊘ v.uToC)
        s += (pol + m1*m2)/ (1+m1*m2*pol)
    end
    s /= g.N
    OrderParams(m,mt,q0,q1,qt,s,st,mh,mht,zeros(5)...)
end

function shortshow(io::IO, x)
    T = typeof(x)
    print(io, T.name.name, "(", join([string(f, "=", getfield(x, f)) for f in fieldnames(T)], ","), ")")
end
Base.show(io::IO, op::OrderParams) = shortshow(io, op)
Base.show(io::IO, op::ThermFunc) = shortshow(io, op)

function solve(; N::Int=1000, α::Float64=0.6, seedξ::Int=-1, kw...)
    seedξ > 0 && srand(seedξ)
    M = round(Int, α * N)
    ξ = rand([-1.,1.], N, M)

    # ξ = randn(N, M)
    σ = rand([-1,1], M)
    solve(ξ, σ; kw...)
end

function solve(ξ::Matrix, σ::Vector{Int}; maxiters::Int = 10000, ϵ::Float64 = 1e-4,
                method = :reinforcement, #[:reinforcement, :decimation]
                y::Float64 = 0., ystep::Float64= 0.00,
                γ::Float64 = 0.2, γstep::Float64= 0.00,
                β = Inf,
                altsolv::Bool = true, altconv = true,
                seed::Int = -1)

    seed > 0 && srand(seed)
    g = FactorGraph(ξ, σ, β=β)
    initrand!(g)

    reinf = ReinfParams(y, ystep, γ, γstep)
    g.reinf = reinf
    while true
        converge!(g, maxiters=maxiters, ϵ=ϵ, reinf=reinf
            , altsolv=altsolv, altconv=altconv)

        tf = ThermFunc(g)
        op = OrderParams(g)
        println("$tf")
        println("$op")
        update_reinforcement!(reinf) || break
        tf.Σint < 0 && break
    end
    return mags(g), magst(g)
end

end #module Pseudo
