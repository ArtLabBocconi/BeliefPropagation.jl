using MacroUtils
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
    K::Int
    ξ::Matrix{Int}
    σ::Vector{Int}
    allm::Vector{Vector{Float64}}
    allm̂::Vector{Vector{Float64}}
    allh::Vector{Vector{Float64}} # for reinforcement
    allpu::Vector{Vector{Float64}} # pup : p(σ=up) from first layer to second
    allpd::Vector{Vector{Float64}} # pdown : p(σ=up) from second layer to first

    function FactorGraphTAP(ξ::Matrix{Int}, σ::Vector{Int}, K::Int)
        N = size(ξ, 1)
        M = length(σ)
        @assert size(ξ, 2) == M
        println("# N=$N M=$M α=$(M/N)")
        new(N, M, K, ξ, σ, [zeros(N) for k=1:K], [zeros(M) for k=1:K]
            , [zeros(N) for k=1:K], [zeros(M) for k=1:K], [zeros(M) for k=1:K])
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
    for m in g.allm
        m[:] = (2*rand(g.N) - 1)/2
    end
    for m̂ in g.allm̂
        m̂[:] = (2*rand(g.M) - 1)/2
    end
    for pu in g.allpu
        pu[:] = rand(g.M)
    end
    for pd in g.allpd
        pd[:] = rand(g.M)
    end
end

function oneBPiter!(g::FactorGraphTAP, r::Float64=0.)
    @extract g N M K allm allm̂ allh allpd allpu ξ σ

    ## factors update ################
    Ĉtot = zeros(K)
    for k=1:K
        m = allm[k]; m̂=allm̂[k]; pd=allpd[k]; pu=allpu[k]
        Ctot = float(N)
        for i=1:N
            Ctot -= m[i]^2
        end
        for a=1:M
            Mtot = 0.
            for i=1:N
                Mtot += ξ[i,a] * m[i]
            end
            Mtot += -m̂[a]*Ctot
            Hp = H(-Mtot / √Ctot); Hm = 1-Hp
            Gp = G(-Mtot / √Ctot); Gm = Gp
            m̂[a] = 1 / √Ctot * (pd[a]*Gp - (1-pd[a])*Gm) / (pd[a]*Hp + (1-pd[a])*Hm)
            Ĉtot[k] += m̂[a] * (Mtot / Ctot + m̂[a])
            pu[a] = Hp - Hm
        end
    end
    #########################################

    ## update exact factor second layer ###
    K2 = div(K-1, 2)
    expf = Complex128[exp(2π*im*p/K) for p=0:K-1]
    expinv = Complex128[expf[(K + (-K2*p) % K) % K + 1] for p=0:K-1]
    for a=1:M
        X = ones(Complex64, K)
        for p=1:K
            for k=1:K
                X[p] *= (1-allpu[k][a]) + allpu[k][a]*expf[p]
            end
        end
        for k=1:K
            s = 0.
            for p=1:K
                s += expinv[p] * X[p] / ((1-allpu[k][a]) + allpu[k][a]*expf[p])
            end
            allpd[k][a] = 0.5*(1-real(s))
            allpd[k][a] += σ[a] == 1 ? real(s) : 0
        end
    end
    #########################################

    ## variables update  #####################
    Δ = 0.
    for k=1:K
        m = allm[k]; m̂=allm̂[k]; h=allh[k]
        for i=1:N
            Mtot = 0.
            for a=1:M
                Mtot += ξ[i, a]* m̂[a]
            end
            h[i] = Mtot + m[i] * Ĉtot[k] + r*h[i]
            oldm = m[i]
            m[i] = tanh(h[i])
            Δ = max(Δ, abs(m[i] - oldm))
        end
    end
    #########################################

    Δ
end

function update_reinforcement!(reinfpar::ReinfParams)
    if reinfpar.wait_count < 10
        reinfpar.wait_count += 1
    else
        reinfpar.r = 1 - (1-reinfpar.r) * (1-reinfpar.r_step)
    end
end

getW(mags::Vector{Vector{Float64}}) = [Int[1-2signbit(m) for m in magk] for magk in mags]

function converge!(g::FactorGraphTAP; maxiters::Int = 10000, ϵ::Float64=1e-5
                                , alt_when_solved::Bool=false, alt_when_converged = false
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
        if alt_when_converged && Δ < ϵ
            println("Converged!")
            break
        end
    end
end

function energy(g::FactorGraphTAP, W::Vector{Vector{Int}})
    @extract g M σ ξ K
    E = 0
    for a=1:M
        σks = Int[ifelse(dot(ξ[:,a], W[k]) > 0, 1, -1) for k=1:K]
        E += σ[a] * sum(σks) > 0 ? 0 : 1
    end
    E
end

energy(g::FactorGraphTAP) = energy(g, getW(mags(g)))

mag(g::FactorGraphTAP, k::Integer, i::Integer) = g.allm[k][i]
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

mags(g::FactorGraphTAP) = g.allm
# mags_noreinf(g::FactorGraphTAP) = Float64[mag_noreinf(v) for v in g.vnodes]


function solve(; N::Int=1000, α::Float64=0.6, K::Int=3, seed_ξ::Int=-1, kw...)
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
                alt_when_converged::Bool = true,
                seed::Int = -1)
    @assert K % 2 == 1
    seed > 0 && srand(seed)
    g = FactorGraphTAP(ξ, σ, K)
    initrand!(g)

    # if method == :reinforcement
    reinfpar = ReinfParams(r, r_step, γ, γ_step)
    converge!(g, maxiters=maxiters, ϵ=ϵ, reinfpar=reinfpar,
            alt_when_solved=alt_when_solved, alt_when_converged=alt_when_converged)
    return getW(mags(g))
end
