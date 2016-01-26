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

## Integration routines #####
## Gauss integration
nint = 100
let s = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
    global gw
    gw(n::Int) = get!(s, n) do
        (x,w) = gausshermite(n)
        return (map(Float64, x * √big(2.0)), map(Float64, w / √(big(π))))
    end
end
function ∫D(f; n=nint, int=nothing)
    (xs, ws) = gw(n)
    s = 0.0
    for (x,w) in zip(xs, ws)
        y = f(x)
        s += w * ifelse(isfinite(y), y, 0.0)
    end
    return s
end

## quadgk integration
# nint = 100
# const ∞ = 30.0
# interval = map(x->sign(x)*abs(x)^2, -1:1/nint:1) .* ∞
#
# ∫D(f) = quadgk(z->begin
#         r = G(z) * f(z)
#         isfinite(r) ? r : 0.0
#     end, interval..., abstol=1e-14,  maxevals=10^10)[1]
######################

type FactorGraphTAP
    N::Int
    M::Int
    ξ::Matrix{Int}
    σ::Vector{Int}
    m::Vector{Float64}
    ρ::Vector{Float64}
    mh::Vector{Float64}
    ρh::Vector{Float64}
    γ::Float64
    y::Float64


    function FactorGraphTAP(ξ::Matrix{Int}, σ::Vector{Int}, γ::Float64, y::Float64)
        N = size(ξ, 1)
        M = length(σ)
        @assert size(ξ, 2) == M
        println("# N=$N M=$M α=$(M/N)")
        new(N, M, ξ, σ, zeros(N), zeros(N), zeros(M), zeros(M),  γ, y)
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
    g.ρ[:] = g.m[:].^2 + 1e-2
    g.mh[:] = (2*rand(g.M) - 1)/2
    g.ρh[:] = g.mh[:].^2 + 1e-2
end

### Update functions
J0(a, b, y) = ∫D(z->H(-a-b*z)^y)
J1(a, b, y) = ∫D(z->H(-a-b*z)^y * GH(-a-b*z))
J2(a, b, y) = ∫D(z->H(-a-b*z)^y * GH(-a-b*z)^2)
J012(a, b, y) = J0(a, b, y), J1(a, b, y), J2(a, b, y)
K0(a, b, γ, y) = K0p(a+γ, b, y) + K0p(a-γ, b, y)
K1(a, b, γ, y) = K1p(a+γ, b, y) + K1p(a-γ, b, y)
K2(a, b, γ, y) = K2p(a+γ, b, y) + K2p(a-γ, b, y)
K0p(a, b, y) = ∫D(z->cosh(a+b*z)^y)
K1p(a, b, y) = ∫D(z->cosh(a+b*z)^y * tanh(a+b*z))
K2p(a, b, y) = ∫D(z->cosh(a+b*z)^y * tanh(a+b*z)^2)
K012(a, b, γ, y) = K0(a, b, γ, y), K1(a, b, γ, y), K2(a, b, γ, y)

function oneBPiter!(g::FactorGraphTAP)
    @extract g N M m ρ mh ρh ξ σ γ y
    # O and Ô are the Onsager rection terms
    # factors update
    Ctot =0.; Rtot = 0.; R̂tot = 0.; O = 0.; Ô = 0.

    for i=1:N
        Ctot += 1 - m[i]^2
        Rtot += ρ[i] - m[i]^2
        O += y*(ρ[i] - m[i]^2) + 1 - m[i]^2
        # O += 1 - m[i]^2
    end
    # println("@ ", Ctot, " ", Rtot, " ", O)
    @assert Ctot > 0
    @assert Rtot > 0

    for a=1:M
        Mtot = 0.
        for i=1:N
            Mtot += ξ[i,a] * m[i]
        end
        Mtot += -mh[a]*O
        j0, j1, j2 = J012(σ[a]*Mtot/√Ctot, √Rtot/√Ctot, y)
        # println(j0)
        @assert isfinite(j0)
        @assert isfinite(j1)
        @assert isfinite(j2)
        @assert j0 > 0
        @assert j2 > 0
        mh[a] = σ[a]/√Ctot * j1 / j0
        ρh[a] = 1/Ctot * j2 / j0
        @assert isfinite(mh[a])
        @assert isfinite(ρh[a])

        R̂tot += ρh[a] - mh[a]^2
        Ô += y*(ρh[a] - mh[a]^2) - (ρh[a] + 1/(Ctot + Rtot)*(mh[a]*Mtot + (y-1)*ρh[a]*Rtot))
        # Ô +=  - (mh[a]^2 + 1/(Ctot)*(mh[a]*Mtot))
    end
    # println("@ ", R̂tot, " ", Ô)


    Δ = 0.
    # variables update
    for i=1:N
        Mtot = 0.
        for a=1:M
            Mtot += ξ[i, a]* mh[a]
        end
        Mtot += - m[i] * Ô
        k0, k1, k2 = K012(Mtot, √R̂tot, γ, y)
        # println(m[i]," ", k0, " ", k1, " ", k2)
        oldm = m[i]
        m[i] = k1 / k0
        ρ[i] = k2 / k0
        Δ = max(Δ, abs(m[i] - oldm))
        assert(Δ > 0.)
        # println(Δ)
    end
    Δ
end
#
# function update_reinforcement!(reinfpar::ReinfParams)
#     if reinfpar.wait_count < 10
#         reinfpar.wait_count += 1
#     else
#         if reinfpar.γ == 0.
#             reinfpar.r = 1 - (1-reinfpar.r) * (1-reinfpar.r_step)
#         else
#             reinfpar.r *= 1 + reinfpar.r_step
#             reinfpar.γ *= 1 + reinfpar.γ_step
#             reinfpar.tγ = tanh(reinfpar.γ)
#         end
#     end
# end

getW(mags::Vector) = Int[1-2signbit(m) for m in mags]

function converge!(g::FactorGraphTAP; maxiters::Int = 10000, ϵ::Float64=1e-5
                                , alt_when_solved::Bool=false, alt_when_converged = false
                                , reinfpar::ReinfParams=ReinfParams())

    for it=1:maxiters
        write("it=$it ... ")
        Δ = oneBPiter!(g)
        E = energy(g)
        @printf("r=%.3f γ=%.3f  E=%d   \tΔ=%f \n", reinfpar.r, reinfpar.γ, E, Δ)
        # update_reinforcement!(reinfpar)
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
                y::Float64 = 0., y_step::Float64= 0.001,
                γ::Float64 = 0., γ_step::Float64=0.,
                alt_when_solved::Bool = true,
                alt_when_converged::Bool = true,
                n= 100,
                seed::Int = -1)
    global nint = n
    seed > 0 && srand(seed)
    g = FactorGraphTAP(ξ, σ, γ, y)
    initrand!(g)

    # if method == :reinforcement
    reinfpar = ReinfParams(y, y_step, γ, γ_step)
    converge!(g, maxiters=maxiters, ϵ=ϵ, reinfpar=reinfpar
            , alt_when_solved=alt_when_solved, alt_when_converged=alt_when_converged)
    return getW(mags(g))
end
