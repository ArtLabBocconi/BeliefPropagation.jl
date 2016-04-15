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
    mt::Vector{Float64}
    ρ::Vector{Float64}
    ρt::Vector{Float64}
    mh::Vector{Float64}
    ρh::Vector{Float64}
    mht::Vector{Float64}
    ρht::Vector{Float64}
    γ::Float64
    y::Float64

    function FactorGraphTAP(ξ::Matrix{Int}, σ::Vector{Int}, γ::Float64, y::Float64)
        N = size(ξ, 1)
        M = length(σ)
        @assert size(ξ, 2) == M
        println("# N=$N M=$M α=$(M/N)")
        new(N, M, ξ, σ, zeros(N), zeros(N), zeros(N), zeros(N), zeros(M), zeros(M), zeros(M), zeros(M), γ, y)
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
    g.mt[:] = (2*rand(g.N) - 1)/2
    g.m[:] = (2*rand(g.N) - 1)/2
    g.ρt[:] = g.mt[:].* g.m[:]. + 1e-4
    g.ρ[:] = g.m[:].^2 + 1e-2

    g.mth[:] = (2*rand(g.N) - 1)/2
    g.mh[:] = (2*rand(g.N) - 1)/2
    g.ρth[:] = g.mth[:].* g.mh[:]. + 1e-4
    g.ρh[:] = g.mh[:].^2 + 1e-2
end

### Update functions
J00(a, b, c, d, y) = ∫D(z->H(-a-b*z)^y * H(-c-d*z))
J10(a, b, c, d, y) = ∫D(z->H(-a-b*z)^y * H(-c-d*z) * GH(-a-b*z))
J20(a, b, c, d, y) = ∫D(z->H(-a-b*z)^y * H(-c-d*z) * GH(-a-b*z)^2)
J01(a, b, c, d, y) = ∫D(z->H(-a-b*z)^y * H(-c-d*z) * GH(-c-d*z))
J02(a, b, c, d, y) = ∫D(z->H(-a-b*z)^y * H(-c-d*z) * GH(-c-d*z)^2)
J11(a, b, c, d, y) = ∫D(z->H(-a-b*z)^y * H(-c-d*z) * GH(-a-b*z) * GH(-c-d*z))

#TODO: change from here
K00(a, b, γ, y) = K0p(a+γ, b, y) + K0p(a-γ, b, y)
K10(a, b, γ, y) = K1p(a+γ, b, y) + K1p(a-γ, b, y)
K20(a, b, γ, y) = K2p(a+γ, b, y) + K2p(a-γ, b, y)
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
        Ctot += 1 - ρ[i]
        Rtot += ρ[i] - m[i]^2
        # O += y*(ρ[i] - m[i]^2) + 1 - m[i]^2
        O += y*(ρ[i] - m[i]^2) + 1 - ρ[i]
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
        j0 <= 0 && (j0=1e-5)
        j2 <= 0 && (j0=1e-10)
        mh[a] = σ[a]/√Ctot * j1 / j0
        ρh[a] = 1/Ctot * j2 / j0
        @assert isfinite(mh[a])
        @assert isfinite(ρh[a])

        R̂tot += ρh[a] - mh[a]^2
        Ô += y*(ρh[a] - mh[a]^2) - (ρh[a] + 1/(Ctot + Rtot)*(mh[a]*Mtot + (y-1)*ρh[a]*Rtot))
        # Ô +=  - (mh[a]^2 + 1/(Ctot)*(mh[a]*Mtot))
    end
    # println("@ ", R̂tot, " ", Ô)
    R̂tot <= 0. && (R̂tot=1e-10)

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
        print("it=$it ... ")
        Δ = oneBPiter!(g)
        E = energy(g)
        @printf("r=%.3f γ=%.3f  E=%d   \tΔ=%f \n", reinfpar.r, reinfpar.γ, E, Δ)
        update_reinforcement!(reinfpar)
        g.γ=reinfpar.γ; g.y=reinfpar.r
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

### Thermodinamic Functions ##########
type OrderParams
    Ψ::Float64
    Σext::Float64
    Σint::Float64
    S::Float64
    Ẽ::Float64
    q₀::Float64
    q₁::Float64
    q̃::Float64
    q̂₀::Float64
    q̂₁::Float64
    δq::Float64
    δq̂::Float64
    function OrderParams(Ψ, Σext, Σint, S, Ẽ, q₀, q₁, q̃, q̂₀, q̂₁, y)
        new(Ψ, Σext, Σint, S, Ẽ, q₀, q₁, q̃, q̂₀, q̂₁, (q₁ - q₀) * y, (q̂₁ - q̂₀) * y)
    end
end
function shortshow(io::IO, x)
    T = typeof(x)
    print(io, T.name.name, "(", join([string(f, "=", getfield(x, f)) for f in fieldnames(T)], ","), ")")
end
Base.show(io::IO, op::OrderParams) = shortshow(io, op)


function energy(g::FactorGraphTAP, W::Vector{Int})
    @extract g M σ ξ
    E = 0
    for a=1:M
        E += σ[a] * dot(ξ[:,a], W) > 0 ? 0 : 1
    end
    E
end

function therm_functions(g::FactorGraphTAP)
    @extract g N M m ρ mh ρh ξ σ γ y
    q0 = 1/N * dot(m,m)
    q1 = 1/N * sum(ρ)
    q̂0 = dot(mh,mh)
    q̂1 = sum(ρh)
    ψ=0.; Σ=0.

    ## SIMIL UPDATE PART #############
    Ctot =0.; Rtot = 0.; R̂tot = 0.; O = 0.; Ô = 0.
    for i=1:N
        Ctot += 1 - ρ[i]
        Rtot += ρ[i] - m[i]^2
        O += y*(ρ[i] - m[i]^2) + 1 - ρ[i]
    end
    for a=1:M
        Mtot = 0.
        for i=1:N
            Mtot += ξ[i,a] * m[i]
        end
        Mtot += -mh[a]*O
        j0 = J0(σ[a]*Mtot/√Ctot, √Rtot/√Ctot, y)
        R̂tot += ρh[a] - mh[a]^2
        Ô += y*(ρh[a] - mh[a]^2) - (ρh[a] + 1/(Ctot + Rtot)*(mh[a]*Mtot + (y-1)*ρh[a]*Rtot))

        ψ += log(j0)
    end
    for i=1:N
        Mtot = 0.
        for a=1:M
            Mtot += ξ[i, a]* mh[a]
        end
        Mtot += - m[i] * Ô
        k0 = K0(Mtot, √R̂tot, γ, y)
        ψ += log(k0)
    end
    ψ /= N
    ψ += -0.5y*(y-1)* q̂1*q1 + 0.5y^2*q̂0*q0 -0.5y*q̂1
    ψ += y * log(2)
    ################
    OrderParams(ψ, 0., 0., 0., 0., q0, q1, 0., q̂0, q̂1, y)
end

energy(g::FactorGraphTAP) = energy(g, getW(mags(g)))

mag(g::FactorGraphTAP, i::Integer) = g.m[i]

mags(g::FactorGraphTAP) = g.m
# mags_noreinf(g::FactorGraphTAP) = Float64[mag_noreinf(v) for v in g.vnodes]


function solve(; N::Int=1000, α = 0.6, seed_ξ = -1, kw...)
    if seed_ξ > 0
        srand(seed_ξ)
    end
    M = round(Int, α * N)
    ξt = Vector{Int}[rand(-1.0:2.0:1.0, N) for a = 1:M]
    # ξ = rand([-1,1], N, M)
    ξ = hcat(ξt...)
    σ = ones(Int, M)
    solve(ξ, σ; kw...)
end

function solve(ξ::Matrix{Int}, σ::Vector{Int}; maxiters = 10000, ϵ = 1e-4,
                method = :reinforcement, #[:reinforcement, :decimation]
                y = 0., y_step = 0.001,
                γ = 0., γ_step = 0.,
                altsolv::Bool = true,
                altconv::Bool = true,
                n= 100,
                seed::Int = -1)
    global nint = n
    seed > 0 && srand(seed)
    g = FactorGraphTAP(ξ, σ, γ, y)
    initrand!(g)

    # if method == :reinforcement
    reinfpar = ReinfParams(y, y_step, γ, γ_step)
    converge!(g, maxiters=maxiters, ϵ=ϵ, reinfpar=reinfpar
            , altsolv=altsolv, altconv=altconv)
    return g, getW(mags(g))
end
