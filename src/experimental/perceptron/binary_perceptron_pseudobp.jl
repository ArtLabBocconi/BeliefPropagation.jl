using MacroUtils

typealias Mess Float64
typealias P Ptr{Mess}
typealias VMess Vector{Mess}
typealias VPMess Vector{P}

Base.getindex(p::Ptr) = unsafe_load(p)
Base.setindex!{T}(p::Ptr{T}, x::T) = unsafe_store!(p, x)
Base.show(io::IO, p::Ptr) = show(io, p[])
Base.show(p::Ptr) = show(p[])

getref(v::Vector, i::Integer) = pointer(v, i)
Mess() = Mess(0.)

G(x) = e^(-(x^2)/2) / √(convert(typeof(x),2) * π)
H(x) = erfc(x / √convert(typeof(x),2)) / 2
#GH(x) = ifelse(x > 30.0, x+(1-2/x^2)/x, G(x) / H(x))
function GHapp(x)
    # print("ghapp")
    y = 1/x
    y2 = y^2
    x + y * (1 - 2y2 * (1 - 5y2 * (1 - 7.4y2)))
end
GH(x) = x > 30.0 ? GHapp(x) : G(x) / H(x)

G(x, β) = (1 - exp(-β)) * G(x)
H(x,β) = (eb=exp(-β); eb + (1-eb)*H(x))
GH(x, β) = β == Inf ? GH(x) : GHapp(x, β) #x > 30.0 ? GHapp(x, β) : G(x, β) / H(x, β)
# function GHapp(x, β)
#     # print("ghapp")
#     # NOTE: not a very good approximation when x is large and β is not
#     y = 1/x
#     y2 = y^2
#     a = e^(-β + (x^2)/2) / ((1 - e^(-β)) * √(2π))
#     return x / (x * a + 1 - y2 * (1 - 3y2 * (1 - 5y2)))
# end
GHapp(x, β) = exp(log(G(x, β)) - log(H(x, β)))
type Fact
    m::VMess
    mt::VMess
    mh::VPMess
    mht::VPMess
    ξ::Vector{Float64}
    σ::Int
end

Fact(ξ, σ) = Fact(VMess(),VMess(), VPMess(), VPMess(), ξ, σ)

type Var
    mh::VMess
    mht::VMess
    m::VPMess
    mt::VPMess

    h::Float64
    ht::Float64
end

Var() = Var(VMess(), VMess(), VPMess(), VPMess(), 0., 0.)

type FactorGraph
    N::Int
    M::Int
    β::Float64
    ξ::Matrix{Float64}
    σ::Vector{Int}
    fnodes::Vector{Fact}
    vnodes::Vector{Var}

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

        new(N, M, β, ξ, σ, fnodes, vnodes)
    end
end

type ReinfParams
    r::Float64
    rstep::Float64
    γ::Float64
    γstep::Float64
    wait_count::Int
    ReinfParams(r,rstep,γ,γstep) = new(r, rstep, γ, γstep, 0)
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

function update!(f::Fact, β)
    @extract f m mh mt mht σ ξ
    M = 0.
    C = 0.

    ## Update mh
    for i=1:deg(f)
        M += ξ[i]*m[i]
        C += ξ[i]^2*(1-m[i]^2)
    end
    for i=1:deg(f)
        Mcav = M - ξ[i]*m[i]
        Ccav = C - ξ[i]^2*(1-m[i]^2)
        Ccav <= 0. && (print("*"); Ccav =1e-5)
        sqC = sqrt(Ccav)
        x = σ*Mcav / sqC
        gh = GH(-x, β)
        @assert isfinite(gh)
        mh[i][] = σ*ξ[i]/sqC * gh
        @assert isfinite(mh[i][])
    end

    M = 0.
    C = 0.
    ## Update mht
    for i=1:deg(f)
        M += ξ[i]*mt[i]
        C += ξ[i]^2*(1-mt[i]^2)
    end
    for i=1:deg(f)
        Mcav = M - ξ[i]*mt[i]
        Ccav = C - ξ[i]^2*(1-mt[i]^2)
        Ccav <= 0. && (print("*"); Ccav =1e-5)
        sqC = sqrt(Ccav)
        x = σ*Mcav / sqC
        gh = GH(-x, β)
        @assert isfinite(gh)
        mht[i][] = σ*ξ[i]/sqC * gh
        @assert isfinite(mht[i][])
    end
end

function update!(v::Var, r::Float64, γ::Float64)
    @extract v m mh mt mht
    Δ = 0.
    h = sum(mh)
    ht = sum(mht)
    uToCenter = atanh(tanh(γ)*tanh(h))
    uToPeriph = atanh(tanh(γ)*tanh(ht + (r-1)*uToCenter))
    h += uToPeriph
    ht += r*uToCenter

    Δ = max(Δ, abs(h - v.h))
    v.h = h
    Δ = max(Δ, abs(ht - v.ht))
    v.ht = ht
    ### compute cavity fields
    # update m
    for a=1:deg(v)
        m[a][] = tanh(h - mh[a])
    end
    # update mt
    for a=1:deg(v)
        mt[a][] = tanh(ht - mht[a])
    end

    Δ
end

function oneBPiter!(g::FactorGraph, reinf::ReinfParams)
    Δ = 0.

    for a=randperm(g.M)
        update!(g.fnodes[a], g.β)
    end

    for i=randperm(g.N)
        d = update!(g.vnodes[i], reinf.r, reinf.γ)
        Δ = max(Δ, d)
    end

    Δ
end

function update_reinforcement!(reinf::ReinfParams)
    if reinf.wait_count < 10
        reinf.wait_count += 1
    else
        reinf.r *= 1 + reinf.rstep
        reinf.γ *= 1 + reinf.γstep
    end
end

getW(mags::Vector) = Int[1-2signbit(m) for m in mags]

function converge!(g::FactorGraph; maxiters::Int = 10000, ϵ::Float64=1e-5
                                , altsolv::Bool=false, altconv = true
                                 , reinf::ReinfParams=ReinfParams())

    for it=1:maxiters
        print("it=$it ... ")
        Δ = oneBPiter!(g, reinf)
        E = energy(g)
        Et = energyt(g)
        @printf("r=%.3f γ=%.3f  E(W̃)=%d E(W)=%d   \tΔ=%f \n", reinf.r, reinf.γ, E, Et, Δ)
        update_reinforcement!(reinf)
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

function mag(v::Var)
    m = tanh(v.h)
    @assert isfinite(m)
    return m
end

function magt(v::Var)
    m = tanh(v.ht)
    @assert isfinite(m)
    return m
end

mags(g::FactorGraph) = Float64[mag(v) for v in g.vnodes]
magst(g::FactorGraph) = Float64[magt(v) for v in g.vnodes]

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
                r::Float64 = 0., rstep::Float64= 0.00,
                γ::Float64 = 0.2, γstep::Float64= 0.00,
                β = Inf,
                altsolv::Bool = true, altconv = true,
                seed::Int = -1)

    seed > 0 && srand(seed)
    g = FactorGraph(ξ, σ, β=β)
    initrand!(g)

    # if method == :reinforcement
    reinf = ReinfParams(r, rstep, γ, γstep)
    converge!(g, maxiters=maxiters, ϵ=ϵ, reinf=reinf
            , altsolv=altsolv, altconv=altconv)
    return mags(g), magst(g)
end
