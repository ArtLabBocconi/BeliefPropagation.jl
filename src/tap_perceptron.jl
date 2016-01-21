using MacroUtils


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
    ξ::Matrix{Int}
    σ::Vector{Int}
    m::Vector{Float64}
    m̂::Vector{Float64}
    h::Vector{Float64} # for reinforcement

    function FactorGraphTAP(ξ::Matrix{Int}, σ::Vector{Int})
        N = size(ξ, 1)
        M = length(σ)
        @assert size(ξ, 2) == M
        println("# N=$N M=$M α=$(M/N)")
        new(N, M, ξ, σ, zeros(N), zeros(M), zeros(N))
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
    g.m̂[:] = (2*rand(g.M) - 1)/2
end

function oneBPiter!(g::FactorGraphTAP, γ::Float64=0.)
    @extract g N M m m̂ h ξ σ

    # factors update
    Ctot = float(N)
    for i=1:N
        Ctot -= m[i]^2
    end
    Ĉtot = 0.
    for a=1:M
        Mtot = 0.
        for i=1:N
            Mtot += ξ[i,a] * m[i]
        end
        Mtot += -m̂[a]*Ctot
        m̂[a] = σ[a] / √Ctot * GH(-σ[a]*Mtot / √Ctot)
        Ĉtot += m̂[a] * (Mtot / Ctot + m̂[a])
    end

    Δ = 0.
    # variables update
    for i=1:N
        Mtot = 0.
        for a=1:M
            Mtot += ξ[i, a]* m̂[a]
        end
        h[i] = Mtot + m[i] * Ĉtot + γ*h[i]
        newm = tanh(h[i])
        oldm = m[i]
        m[i] = newm
        Δ = max(Δ, abs(newm - oldm))
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

function converge!(g::FactorGraphTAP; maxiters::Int = 10000, ϵ::Float64=1e-5, alt_when_solved::Bool=false
                                 , reinfpar::ReinfParams=ReinfParams())

    for it=1:maxiters
        write("it=$it ... ")
        Δ = oneBPiter!(g, reinfpar.γ)
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

function energy(g::FactorGraphTAP, W::Vector{Int})
    @extract g M σ ξ
    E = 0
    for a=1:M
        E += σ[a] * dot(ξ[:,a], W) > 0 ? 0 : 1
    end
    E
end

energy(g::FactorGraphTAP) = energy(g, getW(mags(g)))

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
                r::Float64 = 0., r_step::Float64= 0.001,
                γ::Float64 = 0., γ_step::Float64=0.,
                alt_when_solved::Bool = true,
                seed::Int = -1)

    seed > 0 && srand(seed)
    g = FactorGraphTAP(ξ, σ)
    initrand!(g)

    # if method == :reinforcement
    reinfpar = ReinfParams(r, r_step, γ, γ_step)
    converge!(g, maxiters=maxiters, ϵ=ϵ, reinfpar=reinfpar, alt_when_solved=alt_when_solved)
    return getW(mags(g))
end
