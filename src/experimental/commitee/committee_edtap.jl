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
function ∫D(f; n=nint)
    (xs, ws) = gw(n)
    s = 0.0
    for (x,w) in zip(xs, ws)
        y = f(x)
        s += w * ifelse(isfinite(y), y, 0.0)
    end
    return s
end
#####################

type FactorGraphTAP
    N::Int
    M::Int
    K::Int
    ξ::Matrix{Int}
    σ::Vector{Int}

    ##for allx variables allx[x] corresponds to the k-th perceptron

    # variables first layer
    allm::Vector{Vector{Float64}}
    allρ::Vector{Vector{Float64}}

    # factors first layer
    allmh::Vector{Vector{Float64}}
    allρh::Vector{Vector{Float64}}

    # factors first layer -> factor second layer
    alla::Vector{Vector{Float64}} # = Mtot/√Ctot
    allb::Vector{Vector{Float64}} # = √Rtot/√Ctot
    allj0::Vector{Vector{Float64}}  # = J0(Mtot/√Ctot, √Rtot/√Ctot, y)

    # factors second layer -> factor first layer
    allpd::Vector{Vector{Vector{Float64}}} # = pd[k][a][l+1] = P(#σ up = l)

    γ::Float64
    y::Int

    function FactorGraphTAP(ξ::Matrix{Int}, σ::Vector{Int}, K::Int, γ::Float64, y::Int)
        N = size(ξ, 1)
        M = length(σ)
        @assert size(ξ, 2) == M
        println("# N=$N M=$M α=$(M/N)")
        new(N, M, K, ξ, σ,
            [zeros(N) for k=1:K], [zeros(N) for k=1:K],
            [zeros(M) for k=1:K], [zeros(M) for k=1:K],
            [zeros(M) for k=1:K], [zeros(M) for k=1:K], [ones(M) for k=1:K],
            [[ones(y+1)/2^y for a=1:M] for k=1:K],
            γ, y)
    end
end

type ReinfParams
    y::Int
    y_step::Float64
    γ::Float64
    γ_step::Float64
    tγ::Float64
    wait_count::Int
    ReinfParams(r=3, r_step=0., γ=0., γ_step=0.) = new(r, r_step, γ, γ_step, tanh(γ))
end

function initrand!(g::FactorGraphTAP)
    for m in g.allm
        m[:] = 2*rand(g.N) - 1
    end
    for (ρ,m) in zip(g.allρ, g.allm)
        ρ[:] = m.^2 + 1e-4
    end
    for mh in g.allmh
        mh[:] = 2*rand(g.M) - 1
    end
    for (ρh,mh) in zip(g.allρh, g.allmh)
        ρh[:] = mh.^2 + 1e-4
    end
    # for pu in g.allpu
    #     pu[:] = rand(g.M)
    # end
    # for pd in g.allpd
    #     # pd[:] = ones(g.M)
    #     pd[:] = rand(g.M)
    # end
end

### Update functions
argJ0(a, b, y, i) = ∫D(z->H(-a-b*z)^i * H(+a-b*z)^(y-i))
J0(a, b, y, pd) = sum(i->pd[i+1]*binomial(y,i)*
            argJ0(a, b, y, i), 0:y)
J0(a, b, y) = ∫D(z->(H(-a-b*z) + H(+a-b*z))^y)

argJ1(a, b, y, i) = ∫D(z->H(-a-b*z)^i * H(+a-b*z)^(y-i) *
            (i*GH(-a-b*z)  - (y-i)*GH(a-b*z)))
J1(a, b, y, pd) = 1/y*sum(i->pd[i+1]*binomial(y,i)*
            argJ1(a, b, y, i), 0:y)

argJ2(a, b, y, i) = ∫D(z->H(-a-b*z)^i * H(+a-b*z)^(y-i) *
            ((i*GH(-a-b*z)  - (y-i)*GH(a-b*z))^2
            -(i*GH(-a-b*z)^2  + (y-i)*GH(a-b*z)^2)))
J2(a, b, y, pd) = 1/(y*(y-1))*sum(i->pd[i+1]*binomial(y,i)*
             argJ2(a, b, y, i), 0:y)

J012(a, b, y, pd) = J0(a, b, y, pd), J1(a, b, y, pd), J2(a, b, y, pd)
K0(a, b, γ, y) = K0p(a+γ, b, y) + K0p(a-γ, b, y)
K1(a, b, γ, y) = K1p(a+γ, b, y) + K1p(a-γ, b, y)
K2(a, b, γ, y) = K2p(a+γ, b, y) + K2p(a-γ, b, y)
K0p(a, b, y) = ∫D(z->cosh(a+b*z)^y)
K1p(a, b, y) = ∫D(z->cosh(a+b*z)^y * tanh(a+b*z))
K2p(a, b, y) = ∫D(z->cosh(a+b*z)^y * tanh(a+b*z)^2)
K012(a, b, γ, y) = K0(a, b, γ, y), K1(a, b, γ, y), K2(a, b, γ, y)

function fourier_mc_node(as, bs, j0s, y, σ, iters=10000)
    K = length(as)
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
    pd = [zeros(y+1) for k=1:K]
    for it=1:iters
        z = rand(K)
        X = ones(Complex128, K)
        ps = Float64[H(-as[k]-z[k]*bs[k]) for k=1:K]
        qs = Float64[H(as[k]-z[k]*bs[k]) for k=1:K]
        for p=1:K
            for k=1:K
                X[p] *= qs[k] + ps[k]*expf[p]
            end
        end
        for k=1:K
            s0 = Complex128(0.)
            sp = Complex128(0.)
            sm = Complex128(0.)
            for p=1:K
                xp = X[p] / (qs[k] + ps[k]*expf[p])
                s0 += expinv0[p] * xp
                sp += expinvp[p] * xp
                sm += expinvm[p] * xp
            end
            pp = σ > 0 ? real((s0 + sp)/(s0+2sp)) : real(sm/(s0+2sm))
            pm = σ > 0 ? real(sp/(s0+2sp)) : real((s0 + sm)/(s0+2sm))
            for l=0:y
                pd[k][l+1] += pp^l * pm^(y-l)
            end
        end
    end
    jj0 = prod(j0s)
    for k=1:K
        pd[k] /= iters
        pd[k] /= jj0 / j0s[k]
    end
    return pd
end
####################

function oneBPiter!(g::FactorGraphTAP)
    @extract g N M K allm allρ allmh allρh allpd alla allb allj0 ξ σ γ y

    ## factors update ################
    Ĉtot = zeros(K); R̂tot = zeros(K); Ô = zeros(K)
    for k=1:K
        Ctot = 0.; Rtot = 0.; O =0.
        m = allm[k]; mh=allmh[k];
        ρ = allρ[k]; ρh=allρh[k];
        for i=1:N
            Ctot += 1 - ρ[i]
            Rtot += ρ[i] - m[i]^2
            O += y*(ρ[i] - m[i]^2) + 1 - ρ[i]
            # O += y*(ρ[i] - m[i]^2) + 1 - ρ[i]
        end
        for a=1:M
            pd = allpd[k][a]
            Mtot = 0.
            for i=1:N
                Mtot += ξ[i,a] * m[i]
            end
            Mtot += -mh[a]*O
            j0, j1, j2 = J012(Mtot/√Ctot, √Rtot/√Ctot, y, pd)
            @assert isfinite(j0)
            @assert isfinite(j1)
            @assert isfinite(j2)
            j0 <= 0 && (j0=1e-5)
            j2 <= 0 && (j2=1e-10)
            mh[a] = 1 / √Ctot * j1/j0
            ρh[a] = 1/Ctot * j2 / j0
            @assert isfinite(mh[a])
            @assert isfinite(ρh[a])

            R̂tot[k] += ρh[a] - mh[a]^2
            Ô[k] += y*(ρh[a] - mh[a]^2) - (ρh[a] + 1/(Ctot + Rtot)*(mh[a]*Mtot + (y-1)*ρh[a]*Rtot))

            alla[k][a] = Mtot/√Ctot
            allb[k][a] = √Rtot/√Ctot
            allj0[k][a] = J0(Mtot/√Ctot, √Rtot/√Ctot, y)
        end
    end
    #########################################

    ## update exact factor second layer ###
    for a=1:M
        as = Float64[alla[k][a] for k=1:K]
        bs = Float64[allb[k][a] for k=1:K]
        j0s = Float64[allj0[k][a] for k=1:K]
        pd = fourier_mc_node(as, bs, j0s, y, σ[a])
        for k=1:K
            allpd[k][a][:] = pd[k][:]
        end
    end
    #########################################

    ## variables update  #####################
    Δ = 0.
    for k=1:K
        m = allm[k]; mh=allmh[k]; ρ=allρ[k];
        for i=1:N
            Mtot = 0.
            for a=1:M
                Mtot += ξ[i, a]* mh[a]
            end
            Mtot += - m[i] * Ô[k]
            k0, k1, k2 = K012(Mtot, √(R̂tot[k]), γ, y)
            oldm = m[i]
            m[i] = k1 / k0
            ρ[i] = k2 / k0
            Δ = max(Δ, abs(m[i] - oldm))
        end
    end
    #########################################

    Δ
end

function update_reinforcement!(reinfpar::ReinfParams)
    # if reinfpar.wait_count < 10
    #     reinfpar.wait_count += 1
    # else
    #     reinfpar.r = 1 - (1-reinfpar.r) * (1-reinfpar.r_step)
    # end
end

getW(mags::Vector{Vector{Float64}}) = [Int[1-2signbit(m) for m in magk] for magk in mags]
function print_overlaps(W::Vector{Vector{Int}}; meanvar = true)
    K = length(W)
    N = length(W[1])
    q = Float64[]
    for k=1:K
        for p=k+1:K
            push!(q, dot(W[k],W[p])/N)
        end
    end
    if meanvar
        println("overlaps mean,std = ",mean(q), ",", std(q))
    else
        println(q)
    end
end
function converge!(g::FactorGraphTAP; maxiters::Int = 10000, ϵ::Float64=1e-5
                                , altsolv::Bool=false, altconv = false
                                , reinfpar::ReinfParams=ReinfParams())

    for it=1:maxiters
        write("it=$it ... ")
        Δ = oneBPiter!(g)
        W = getW(mags(g))
        E = energy(g, W)
        print_overlaps(W, meanvar=true)
        @printf("y=%.3f γ=%.3f  E=%d   \tΔ=%f \n", reinfpar.y, reinfpar.γ, E, Δ)
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

function energy(g::FactorGraphTAP, W::Vector{Vector{Int}})
    @extract g M σ ξ K
    E = 0
    for a=1:M
        σks = Int[ifelse(dot(ξ[:,a], W[k]) > 0, 1, -1) for k=1:K]
        # println(σks)
        E += σ[a] * sum(σks) > 0 ? 0 : 1
    end
    E
end

energy(g::FactorGraphTAP) = energy(g, getW(mags(g)))

mag(g::FactorGraphTAP, k::Integer, i::Integer) = g.allm[k][i]


mags(g::FactorGraphTAP) = g.allm
# mags_noreinf(g::FactorGraphTAP) = Float64[mag_noreinf(v) for v in g.vnodes]


function solve(; N::Int=1000, α::Float64=0.6, seed_ξ::Int=-1,
                    K::Int = 3, kw...)
    if seed_ξ > 0
        srand(seed_ξ)
    end
    M = round(Int, α * K * N)
    ξ = rand([-1,1], N, M)
    σ = ones(Int, M)
    solve(ξ, σ; K=K, kw...)
end

function solve(ξ::Matrix{Int}, σ::Vector{Int}; maxiters::Int = 10000, ϵ::Float64 = 1e-4,
                method = :reinforcement, #[:reinforcement, :decimation]
                K::Int=3,
                y::Int = 3,
                γ::Float64 = 0., γ_step::Float64=0.,
                altsolv::Bool = true,
                altconv::Bool = false,
                seed::Int = -1)
    @assert K % 2 == 1
    seed > 0 && srand(seed)
    g = FactorGraphTAP(ξ, σ, K, γ, y)
    initrand!(g)

    # if method == :reinforcement
    reinfpar = ReinfParams(y, 0., γ, γ_step)
    converge!(g, maxiters=maxiters, ϵ=ϵ, reinfpar=reinfpar,
            altsolv=altsolv, altconv=altconv)
    return getW(mags(g))
end
