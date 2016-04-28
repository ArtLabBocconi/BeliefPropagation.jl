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
    ρ::VMess
    mh::VPMess
    ρh::VPMess
    ξ::SubArray
    σ::Int
end

Fact(ξ, σ) = Fact(VMess(),VMess(), VPMess(), VPMess(), ξ, σ)

type Var
    mh::VMess
    ρh::VMess
    m::VPMess
    ρ::VPMess
    λ::Float64 #L2 regularization

    #used only in BP+reinforcement
    h1::Mess
    h2::Mess
end

Var() = Var(VMess(), VMess(), VPMess(), VPMess(), 0., Mess(0), Mess(0))

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
        fnodes = [Fact(sub(ξ, :, a), σ[a]) for a=1:M]
        vnodes = [Var() for i=1:N]

        ## Reserve memory in order to avoid invalidation of Refs
        for (a,f) in enumerate(fnodes)
            sizehint!(f.m, N)
            sizehint!(f.ρ, N)
            sizehint!(f.mh, N)
            sizehint!(f.ρh, N)
        end
        for (i,v) in enumerate(vnodes)
            sizehint!(v.m, M)
            sizehint!(v.ρ, M)
            sizehint!(v.mh, M)
            sizehint!(v.ρh, M)
            v.λ = λ
        end

        for i=1:N, a=1:M
            f = fnodes[a]
            v = vnodes[i]

            push!(v.mh, Mess())
            push!(v.ρh, Mess())
            push!(f.mh, getref(v.mh, length(v.mh)))
            push!(f.ρh, getref(v.ρh, length(v.ρh)))

            push!(f.m, Mess())
            push!(f.ρ, Mess())
            push!(v.m, getref(f.m, length(f.m)))
            push!(v.ρ, getref(f.ρ, length(f.ρ)))
        end

        new(N, M, β, ξ, σ, fnodes, vnodes)
    end
end

type ReinfParams
    r::Float64
    rstep::Float64
    γ::Float64
    γstep::Float64
    tγ::Float64
    wait_count::Int
    ReinfParams(r=0.,rstep=0.,γ=0.,γstep=0.) = new(r, rstep, γ, γstep, tanh(γ))
end

deg(f::Fact) = length(f.m)
deg(v::Var) = length(v.m)

function initrand!(g::FactorGraph)
    for f in g.fnodes
        f.m[:] = (2*rand(deg(f)) - 1)/2
        f.ρ[:] = 1e-5
    end
    for v in g.vnodes
        v.mh[:] = (2*rand(deg(v)) - 1)/2
        v.ρh[:] = 1e-5
    end
end

#TODO
function update!(f::Fact, β)
    @extract f m mh ρ ρh σ ξ
    M = 0.
    C = 0.
    for i=1:deg(f)
        M += ξ[i]*m[i]
        C += ξ[i]^2*ρ[i]
    end
    for i=1:deg(f)
        Mcav = M - ξ[i]*m[i]
        Ccav = C - ξ[i]^2*ρ[i]
        Ccav <= 0. && (print("*"); Ccav =1e-5)
        sqrtC = sqrt(Ccav)
        x = σ*Mcav / sqrt(Ccav)
        gh = GH(-x, β)
        @assert isfinite(gh)
        mh[i][] = σ*ξ[i]/sqrtC * gh
        ρh[i][] = (ξ[i]/sqrtC)^2 *(x*gh + gh^2) # -∂^2 log ν(W)
        @assert isfinite(mh[i][])
        @assert isfinite(ρh[i][])
    end
end

function update!(v::Var, r::Float64 = 0.)
    @extract v m mh ρ ρh λ h1 h2
    Δ = 0.

    v.h1 = sum(mh) + r*h1
    v.h2 = λ + sum(ρh) + r*h2
    # @assert v.h2 > 0 "$(v.h2)"
    v.h2<0 && (print("!"); v.h2 = 1e-5)
    ### compute cavity fields
    for a=1:deg(v)
        h1 = v.h1 - mh[a]
        h2 = v.h2 - ρh[a]
        newm = h1 / h2
        oldm = m[a][]
        Δ = max(Δ, abs(newm - oldm))
        m[a][] = newm
        ρ[a][] = 1/h2
    end

    Δ
end

function oneBPiter!(g::FactorGraph, r::Float64=0.)
    Δ = 0.

    for a=randperm(g.M)
        update!(g.fnodes[a], g.β)
    end

    for i=randperm(g.N)
        d = update!(g.vnodes[i], r)
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

getW(mags::Vector) = Int[1-2signbit(m) for m in mags]

function converge!(g::FactorGraph; maxiters::Int = 10000, ϵ::Float64=1e-5
                                , altsolv::Bool=false, altconv = true
                                 , reinfpar::ReinfParams=ReinfParams())

    for it=1:maxiters
        print("it=$it ... ")
        Δ = oneBPiter!(g, reinfpar.r)
        E = energy(g)
        Etrunc = energy_trunc(g)
        @printf("r=%.3f γ=%.3f  E(W=mags)=%d E(trunc W)=%d   \tΔ=%f \n", reinfpar.r, reinfpar.γ, E, Etrunc, Δ)
        update_reinforcement!(reinfpar)
        if altsolv && E == 0
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

energy(g::FactorGraph) = energy(g, mags(g))
energy_trunc(g::FactorGraph) = energy(g, getW(mags(g)))

function mag(v::Var)
    m = v.h1/v.h2
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


# function batch_renorm!(ξ)
#     N, M = size(ξ)
#     μ = zeros(N)
#     σ = zeros(N)
#     for i=1:N
#         μ[i] = mean(ξ[i,:])
#     end
#     for i=1:N
#         μ[i] = mean(ξ[i,:])
#     end
# end


function solve_test(; N::Int=1000, α::Float64=0.6, biasξ = 0., seedξ::Int=-1, kw...)
    seedξ > 0 && srand(seedξ)
    M = round(Int, α * N)
    # ξ = rand([-1.,1.], N, M)
    ξ = randn(N, M)
    # σ = rand([-1,1], M)
    σ = zeros(Int, M)
    for a=1:M
        σ[a] = ξ[1, a] > biasξ ? 1 : -1
    end
    if biasξ != 0
        ξnew = ones(N+1, M)
        ξnew[1:N, 1:M] = ξ
        ξ = ξnew
    end

    solve(ξ, σ; kw...)
end

function solve(; N::Int=1000, α::Float64=0.6, biasξ = 0., seedξ::Int=-1, kw...)
    seedξ > 0 && srand(seedξ)
    M = round(Int, α * N)
    ξ = rand([-1.,1.], N, M) + biasξ
    if biasξ != 0
        ξnew = ones(N+1, M)
        ξnew[1:N, 1:M] = ξ
        ξ = ξnew
    end

    # ξ = randn(N, M)
    σ = rand([-1,1], M)
    solve(ξ, σ; kw...)
end

function solve(ξ::Matrix, σ::Vector{Int}; maxiters::Int = 10000, ϵ::Float64 = 1e-4,
                method = :reinforcement, #[:reinforcement, :decimation]
                r::Float64 = 0., rstep::Float64= 0.001,
                λ::Float64 = 1., β = Inf,
                altsolv::Bool = true, altconv = true,
                seed::Int = -1)

    seed > 0 && srand(seed)
    g = FactorGraph(ξ, σ, λ, β=β)
    initrand!(g)

    # if method == :reinforcement
    reinfpar = ReinfParams(r, rstep)
    converge!(g, maxiters=maxiters, ϵ=ϵ, reinfpar=reinfpar
            , altsolv=altsolv, altconv=altconv)
    return mags(g)
end
