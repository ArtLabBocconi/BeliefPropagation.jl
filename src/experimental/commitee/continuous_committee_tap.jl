using ExtractMacro
using FastGaussQuadrature

G(x) = e^(-(x^2)/2) / √(convert(typeof(x),2) * π)
H(x) = erfc(x / √convert(typeof(x),2)) / 2
#GH(x) = ifelse(x > 30.0, x+(1-2/x^2)/x, G(x) / H(x))
function GHapp(x)
    y = 1/x
    y2 = y^2
    x + y * (1 - 2y2 * (1 - 5y2 * (1 - 7.4y2)))
end
GH(x) = x > 30.0 ? GHapp(x) : G(x) / H(x)

type FactorGraphTAP
    N::Int
    M::Int
    ξ::Matrix
    ξ2::Matrix
    σ::Vector{Int}
    m::Matrix{Float64}
    ρ::Matrix{Float64}
    mhdown::Vector{Matrix{Float64}}
    ρhdown::Vector{Matrix{Float64}}
    mhup::Vector{Matrix{Float64}}
    ρhup::Vector{Matrix{Float64}}

    λ::Float64 #L2 regularizer
    h1::Matrix{Float64}} # for reinforcement
    h2::Matrix{Float64}} # for reinforcement
    function FactorGraphTAP(ξ::Matrix, σ::Vector{Int}, K::Int, λ::Float64=1)
        N = size(ξ, 1)
        M = length(σ)
        @assert size(ξ, 2) == M
        println("# N=$N M=$M α=$(M/N)")
        new(N, M, ξ, ξ.^2, σ, zeros(N), zeros(N), zeros(M), zeros(M)
            , λ, zeros(N), zeros(N))
    end
end

type ReinfParams
    r::Float64
    r_step::Float64
    γ::Float64
    γ_step::Float64
    tγ::Float64
    wait_count::Int
    ReinfParams(r=0., r_step=0., γ=0., γ_step=0.) = new(r, r_step, γ, γ_step, tanh(γ))
end

function initrand!(g::FactorGraphTAP)
    g.m[:] = (2*rand(g.N) - 1)/2
    g.ρ[:] = 1e-5
    g.mh[:] = (2*rand(g.M) - 1)/2
    g.ρh[:] = 1e-5
end

function oneBPiter!(g::FactorGraphTAP, r::Float64=0.)
    @extract g N M m ρ mh ρh h1 h2 ξ σ λ

    # factors update
    # Ĉtot = 0.
    for a=1:M
        Mtot = 0.; Ctot = 0.
        for i=1:N
            Ctot += ξ[i,a]^2*ρ[i]
            Mtot += ξ[i,a] * m[i]
        end
        Mtot += -mh[a]*Ctot
        x = σ[a]*Mtot / sqrt(Ctot)
        gh = GH(-x)
        mh[a] = σ[a]/ sqrt(Ctot) * gh
        ρh[a] = 1/Ctot *(x*gh + gh^2)
    end

    Δ = 0.
    # variables update
    for i=1:N
        Mtot = 0.
        Ctot = 0.
        for a=1:M
            Mtot += ξ[i, a]* mh[a]
            Ctot += ξ[i, a]^2* ρh[a]
        end
        h1[i] = Mtot + m[i] * Ctot + r*h1[i]
        h2[i] = λ + Ctot + r*h2[i]
        oldm = m[i]
        m[i] = h1[i]/h2[i]
        ρ[i] = 1/h2[i]
        Δ = max(Δ, abs(m[i] - oldm))
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

function converge!(g::FactorGraphTAP; maxiters::Int = 10000, ϵ::Float64=1e-5
                                , altsolv::Bool=false, altconv = false
                                , reinfpar::ReinfParams=ReinfParams())

    for it=1:maxiters
        write("it=$it ... ")
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

function energy(g::FactorGraphTAP, W::Vector)
    @extract g M σ ξ
    E = 0
    for a=1:M
        E += σ[a] * dot(ξ[:,a], W) > 0 ? 0 : 1
    end
    E
end

energy(g::FactorGraphTAP) = energy(g, mags(g))
energy_trunc(g::FactorGraphTAP) = energy(g, getW(mags(g)))

mag(g::FactorGraphTAP, i::Integer) = g.m[i]
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

mags(g::FactorGraphTAP) = g.m
# mags_noreinf(g::FactorGraphTAP) = Float64[mag_noreinf(v) for v in g.vnodes]

function solve(; N::Int=201, K::Int =11, α::Float64=0.6, seed_ξ::Int=-1, kw...)
    if seed_ξ > 0
        srand(seed_ξ)
    end
    M = round(Int, α * K * N)
    ξ = rand([-1.,1.], N, M)
    σ = ones(Int, M)
    solve(ξ, σ; K=K, kw...)
end

function solve(ξ::Matrix, σ::Vector{Int}; maxiters::Int = 10000, ϵ::Float64 = 1e-4,
                method = :reinforcement, #[:reinforcement, :decimation]
                r::Float64 = 0., r_step::Float64= 0.001,
                λ::Float64 = 1., # L2 regularizer
                altsolv::Bool = true,
                K::Int = 11,
                seed::Int = -1)

    seed > 0 && srand(seed)
    g = FactorGraphTAP(ξ, σ, K, λ)
    initrand!(g)

    # if method == :reinforcement
    reinfpar = ReinfParams(r, r_step, 0. , 0.)
    converge!(g, maxiters=maxiters, ϵ=ϵ, reinfpar=reinfpar, altsolv=altsolv)
    return mags(g)
end
