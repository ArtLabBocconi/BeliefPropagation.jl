module PF
using MacroUtils
using FastGaussQuadrature
using JLD

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
let s = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
    global gw
    gw(n::Int) = get!(s, n) do
        (x,w) = gausshermite(n)
        return (map(Float64, x * √big(2.0)), map(Float64, w / √(big(π))))
    end
end

nint=100 #change by solve
function ∫D(f; n=nint)
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

type ThermFunc
    ϕ::Float64
    Σext::Float64
    Σint::Float64
    E::Float64
    Ẽ::Float64
end
ThermFunc()=ThermFunc(zeros(5)...)

function shortshow(io::IO, x)
    T = typeof(x)
    print(io, T.name.name, "(", join([string(f, "=", getfield(x, f)) for f in fieldnames(T)], ","), ")")
end
Base.show(io::IO, op::OrderParams) = shortshow(io, op)

function veryshortshow(io::IO, x)
	T = typeof(x)
	print(io, join([string(getfield(x, f)) for f in fieldnames(T)], " "))
end

function reset!(tf::ThermFunc)
    T = typeof(tf)
    for f in fieldnames(T)
        setfield!(tf, f, 0.)
    end
end

type FactorGraphTAP
    N::Int
    M::Int
    ξ::Matrix{Float64}
    σ::Vector{Int}
    m::Vector{Float64}
    mt::Vector{Float64}
    ρ::Vector{Float64}
    ρt::Vector{Float64}
    mh::Vector{Float64}
    ρh::Vector{Float64}
    mth::Vector{Float64}
    ρth::Vector{Float64}
    γ::Float64
    y::Float64

    O::Vector{Float64}
    Õ::Vector{Float64}
    Oh::Vector{Float64}
    Õh::Vector{Float64}

    op::OrderParams
    tf::ThermFunc

    function FactorGraphTAP(ξ::Matrix{Float64}, σ::Vector{Int}, γ::Float64, y::Float64)
        N = size(ξ, 1)
        M = length(σ)
        @assert size(ξ, 2) == M
        println("# N=$N M=$M α=$(M/N)")
        new(N, M, ξ, σ, zeros(N), zeros(N), zeros(N), zeros(N), zeros(M), zeros(M), zeros(M), zeros(M)
            , γ, y, zeros(2), zeros(2), zeros(2), zeros(2)
            , OrderParams(), ThermFunc())
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
    g.m[:] = g.mt[:] + 1e-3*(2*rand(g.N) - 1)
    g.ρ[:] = g.m[:].^2 + 1e-2
    g.ρt[:] = g.mt[:].* g.m[:] + 1e-4

    g.mth[:] = (2*rand(g.M) - 1)/2
    g.mh[:] = g.mth[:] + 1e-3*(2*rand(g.M) - 1)
    g.ρh[:] = g.mh[:].^2  + 1e-2
    g.ρth[:] = g.mth[:].* g.mh[:] + 1e-4
end

### Update functions
# J00(a, b, c, d, y) = y == 0 ? H(-c/√(1+d^2)) : ∫D(z->H(-a-b*z)^y * H(-c-d*z))
J00(a, b, c, d, y) = ∫D(z->H(-a-b*z)^y * H(-c-d*z))
J10(a, b, c, d, y) = ∫D(z->H(-a-b*z)^y * H(-c-d*z) * GH(-a-b*z)) #TODO version a y=0
J20(a, b, c, d, y) = ∫D(z->H(-a-b*z)^y * H(-c-d*z) * GH(-a-b*z)^2)
J01(a, b, c, d, y) = ∫D(z->H(-a-b*z)^y * H(-c-d*z) * GH(-c-d*z)) #TODO version a y=0
J02(a, b, c, d, y) = ∫D(z->H(-a-b*z)^y * H(-c-d*z) * GH(-c-d*z)^2)
J11(a, b, c, d, y) = ∫D(z->H(-a-b*z)^y * H(-c-d*z) * GH(-a-b*z) * GH(-c-d*z))
JD0(a, b, c, d, y) = ∫D(z->H(-a-b*z)^y * H(-c-d*z) * (-GH(-a-b*z)*(a+b*z)-GH(-a-b*z)^2))
J0D(a, b, c, d, y) = ∫D(z->H(-a-b*z)^y * H(-c-d*z) * (-GH(-c-d*z)*(c+d*z)-GH(-c-d*z)^2))
JY(a, b, c, d, y) = ∫D(z->H(-a-b*z)^y * H(-c-d*z) * log(H(-a-b*z)))

K00(a, b, c, γ, y) = exp(c)*K0p(a+γ, b, y) + exp(-c)*K0p(a-γ, b, y)
K10(a, b, c, γ, y) = exp(c)*K1p(a+γ, b, y) + exp(-c)*K1p(a-γ, b, y)
K20(a, b, c, γ, y) = exp(c)*K2p(a+γ, b, y) + exp(-c)*K2p(a-γ, b, y)
K11(a, b, c, γ, y) = exp(c)*K1p(a+γ, b, y) - exp(-c)*K1p(a-γ, b, y)
K01(a, b, c, γ, y) = exp(c)*K0p(a+γ, b, y) - exp(-c)*K0p(a-γ, b, y)
KY(a, b, c, γ, y) = exp(c)*KYp(a+γ, b, y) + exp(-c)*KYp(a-γ, b, y)
# K0p(a, b, y) = y == 0 ? 1. : ∫D(z->cosh(a+b*z)^y)
K0p(a, b, y) = ∫D(z->cosh(a+b*z)^y)
K1p(a, b, y) = ∫D(z->cosh(a+b*z)^y * tanh(a+b*z))
K2p(a, b, y) = ∫D(z->cosh(a+b*z)^y * tanh(a+b*z)^2)
KYp(a, b, y) = ∫D(z->cosh(a+b*z)^y * log(cosh(a+b*z)))

function thermfun(g::FactorGraphTAP)
    @extract g : N M m ρ mh ρh mt ρt mth ρth ξ σ γ y op tf Oh Õh O Õ
    @extract op : q̂t q̂0 q̂1 ŝ ŝt qt q0 q1 s st
    Δϕ = 0.; ΔΣint=0.
    ϕ = 0.; Σint=0;
    # for a=1:M
    #     rh = [y*(ρh[a]-mh[a]^2)+gg[a], ρth[a]-mh[a]*mth[a]]
    #     r̃h = [y*(ρth[a]-mh[a]*mth[a]), g̃g̃[a] - mth[a]^2]
    #     # println("Oh=$Oh")
    #     # println("Õh=$Õh")
    #     for i=1:N
    #         ϵ = -(ξ[i, a]* mh[a] - m[i]*rh[1] - mt[i]*rh[2])
    #         ϵt = -(ξ[i, a]* mth[a] - m[i]*r̃h[1] - mt[i]*r̃h[2])
    #
    #         Δϕ += ϵt*mt[i] + ϵ*y*m[i] + 0.5*ϵt^2*(1-mt[i]^2)
    #         Δϕ += 0.5*ϵ^2*(y*(1-ρ[i]) + y^2*(ρ[i]-m[i]^2))
    #         Δϕ += y*ϵt*ϵ*(ρt[i] - mt[i]*m[i])
    #
    #         ΔΣint += ϵ*m[i]
    #         ΔΣint += 0.5*ϵ^2*((1-ρ[i]) + 2y*(ρ[i]-m[i]^2))
    #         ΔΣint += ϵt*ϵ*(ρt[i] - mt[i]*m[i])
    #     end
    # end
    #TODO fore posso levarlo
    q0 = 0.;q1=0.;qt=0.; s=0.;st=0.; mm=0.;mmt=0.;
    for i=1:N
        mm += m[i]
        mmt += mt[i]
        q0 += m[i]*m[i]
        q1 += ρ[i]
        qt += mt[i]^2
        s += ρt[i]
        st += m[i]*mt[i]
    end
    q1<q0 && (q1=q0; print("!nh "))
    p = (s-st)^2/(q1-q0)
    # @assert (q1-q0)*(N-qt)-(s-st)^2 >= 0 "(q1-q0)*(1-qt)-(s-st)^2 > 0 q0=$q0 q1=$q1 qt=$qt s=$s st=$st"
    # @assert q1>=q0 "q1>q0 q0=$q0 q1=$q1"
    !isfinite(p) && (p=0.; print("!p1 "))
    p >= N-qt && (p=N-qt-1e-3; print("!p1 "))

    O[:] = [y*(q1 - q0) + N - q1, s - st] #Onsager Reaction on m, coefficients of mh and mth
    Õ[:] = [y*(s - st), N - qt]           #Onsager Reaction on mt, coefficients of mh and mth

    gg = zeros(M); g̃g̃ = zeros(M)
    for a=1:M
        Mtot = 0.
        M̃tot = 0.
        for i=1:N
            # @assert isapprox(m[i], mt[i], atol=1e-4) "ass i=$i m[i] mt[i],  $(m[i]) $(mt[i])"
            Mtot += ξ[i,a] * m[i]
            M̃tot += ξ[i,a] * mt[i]
        end
        Mtot += (-mh[a]*O[1] - mth[a]*O[2])
        M̃tot += (-mh[a]*Õ[1] - mth[a]*Õ[2])

        den1 = √(N-q1)
        den2 = √((N-qt)-p)
        coeffs = (σ[a]*Mtot/den1, √(q1-q0)/den1,
                    σ[a]*M̃tot/den2, √p/den2)
        j00 = J00(coeffs..., y)
        j01 = J01(coeffs..., y)
        j02 = J02(coeffs..., y)
        jd0 = JD0(coeffs..., y)
        j0d = J0D(coeffs..., y)

        ϕ += log(j00)
        # Σint += JY(coeffs..., y)/j00

        # @assert isapprox(mth[a], mh[a], atol=1e-4) "mh[a] mth[a],  $(mh[a]) $(mth[a])"
        gg[a] = 1/den1^2 * jd0 / j00
        g̃g̃[a] = 1/den2^2 * (j02 / j00 + j0d / j00)
        # a==1 &&(println("g̃g̃[a]=$(g̃g̃[a]) j02=$j02 j0d=$j0d j00=$j00"))

        @assert isfinite(mh[a])
        @assert isfinite(ρh[a])
    end

    for i=1:N
        Mtot = 0.
        M̃tot = 0.
        for a=1:M
            Mtot += ξ[i, a]* mh[a]
            M̃tot += ξ[i, a]* mth[a]
        end
        Mtot += (-m[i]*Oh[1] - mt[i]*Oh[2])
        M̃tot += (-m[i]*Õh[1] - mt[i]*Õh[2])

        γeff = ŝ - ŝt + γ
        coeffs = (Mtot, √(q̂1-q̂0), M̃tot, γeff)
        k00 = K00(coeffs..., y)
        ϕ0 = log(k00)
        ϕ += ϕ0
        # Σint0 = KY(coeffs..., y) / k00

        for a=1:M
            rh = [y*(ρh[a]-mh[a]^2)+gg[a], ρth[a]-mh[a]*mth[a]]
            r̃h = [y*(ρth[a]-mh[a]*mth[a]), g̃g̃[a] - mth[a]^2]

            ϵ = ξ[i, a]* mh[a] - m[i]*rh[1] - mt[i]*rh[2]
            ϵt = ξ[i, a]* mth[a] - m[i]*r̃h[1] - mt[i]*r̃h[2]
            ϵs = ρth[a] - mth[a]*mh[a]
            ϵq = ρh[a] - mh[a]^2
            Mcav = Mtot - ϵ
            M̃cav = M̃tot - ϵt
            γcav = γeff - ϵs
            Qcav = q̂1-q̂0 - ϵq
            Qcav < 0 && (Qcav=0.)
            coeffs = (Mcav, √Qcav, M̃cav, γcav)
            k00 = K00(coeffs..., y)
            # pritnln("here")
            # δ1 = (ϵt*mt[i]  + 0.5*ϵt^2*(1-mt[i]^2))
            #     #  + ϵ*y*m[i] + 0.5*ϵ^2*(y*(1-ρ[i]) + y^2*(ρ[i]-m[i]^2))
            #     #  + y*ϵt*ϵ*(ρt[i] - mt[i]*m[i]))
            # δ2 =  log(k00) - ϕ0
            # @assert isapprox(δ1, δ2, atol=1e-4) "$δ1 $δ2"
            Δϕ += log(k00) - ϕ0
            # ΔΣint += KY(coeffs..., y)/k00 - Σint0
        end

    end
    ϕ = (ϕ + Δϕ) / N
    ϕ += - y*γ*s + y*log(2)

    # tf.ϕ /= N
    # tf.ϕ += -0.5y*(y-1)* op.q̂1*q1 + 0.5y^2*q̂0*q0 - 0.5y*q̂1 + 0.5q̂t*qt - 0.5q̂t
    # tf.ϕ += -y*(s*ŝ - st*ŝt) + y * log(2) #- y*m̂*mm -m̂t*mmt
    # tf.Σint /= N
    # tf.Σint += -(y-0.5)* q̂1*q1 + y*q̂0*q0 -0.5q̂1 +log(2)
    # tf.Σint += -(s*ŝ - st*ŝt) - γ*s # - m̂*mm

    return ϕ
end

function compute_tf!(g::FactorGraphTAP)
    @extract g : tf
    ϕ = thermfun(g)

    # δ = 1e-2
    # yold = g.y
    # g.y += δ
    # for it=1:20
    #     Δ = oneBPiter!(g, false, dump=0.)
    #     print("it=$it  y=$(g.y)   \tΔ=$Δ ...")
    #     E = energy(g)
    #     println(g.op)
    #     Δ < 1e-7 && break
    # end
    # ϕnew = thermfun(g)
    # println(ϕnew)
    # g.y = yold

    println(tf)
    reset!(tf)
    tf.ϕ = ϕ
    # tf.Σint = (ϕnew - ϕ) / δ
end

function oneBPiter!(g::FactorGraphTAP, fixmt::Bool; dump=0.
        , computetf = false)
    @extract g N M m ρ mh ρh mt ρt mth ρth ξ σ γ y op tf O Õ Oh Õh
    reset!(tf)
    # O and Ô are the Onsager rection terms
    q0 = 0.;q1=0.;qt=0.; s=0.;st=0.; mm=0.;mmt=0.;
    gg = zeros(M)
    g̃g̃ = zeros(M)
    Δϕ = 0.
    for i=1:N
        mm += m[i]
        mmt += mt[i]
        q0 += m[i]*m[i]
        q1 += ρ[i]
        qt += mt[i]^2
        s += ρt[i]
        st += m[i]*mt[i]
    end
    q1<q0 && (q1=q0; print("!nh "))
    p = (s-st)^2/(q1-q0)
    # @assert (q1-q0)*(N-qt)-(s-st)^2 >= 0 "(q1-q0)*(1-qt)-(s-st)^2 > 0 q0=$q0 q1=$q1 qt=$qt s=$s st=$st"
    # @assert q1>=q0 "q1>q0 q0=$q0 q1=$q1"
    !isfinite(p) && (p=0.; print("!p1 "))
    p >= N-qt && (p=N-qt-1e-3; print("!p1 "))

    # println("q0=$(q0/N) q1=$(q1/N) qt=$(qt/N) s=$(s/N) st=$(st/N)")
    # if q1>=q0
    #     print("!")
    #     # q1 = q0 + 1e-4
    #     q1 = q0
        # @assert q1>=q0 "q1>q0 q0=$q0 q1=$q1"
    # end
    O[:] = [y*(q1 - q0) + N - q1, s - st] #Onsager Reaction on m, coefficients of mh and mth
    Õ[:] = [y*(s - st), N - qt]           #Onsager Reaction on mt, coefficients of mh and mth

    Oh[:] = [0.,0.]       #Onsager Reaction on mh, coefficients of m and mt
    Õh[:] = [0.,0.]       #Onsager Reaction on mth, coefficients of m and mt
    for a=1:M
        Mtot = 0.
        M̃tot = 0.
        for i=1:N
            # @assert isapprox(m[i], mt[i], atol=1e-4) "ass i=$i m[i] mt[i],  $(m[i]) $(mt[i])"
            Mtot += ξ[i,a] * m[i]
            M̃tot += ξ[i,a] * mt[i]
        end
        Mtot += (-mh[a]*O[1] - mth[a]*O[2])
        M̃tot += (-mh[a]*Õ[1] - mth[a]*Õ[2])

        den1 = √(N-q1)
        den2 = √((N-qt)-p)
        coeffs = (σ[a]*Mtot/den1, √(q1-q0)/den1,
                    σ[a]*M̃tot/den2, √p/den2)
        j00 = J00(coeffs..., y)
        j10 = J10(coeffs..., y)
        j20 = J20(coeffs..., y)
        j01 = J01(coeffs..., y)
        j02 = J02(coeffs..., y)
        j11 = J11(coeffs..., y)
        jd0 = JD0(coeffs..., y)
        j0d = J0D(coeffs..., y)
        mh[a] = σ[a]/den1 * j10 / j00
        ρh[a] = 1/den1^2 * j20 / j00
        if !fixmt
            mth[a] = σ[a]/den2 * j01 / j00
        end
        ρth[a] = 1/(den2*den1) * j11 / j00

        tf.ϕ += log(j00)
        # tf.Σint += JY(coeffs..., y)/j00

        # # @assert isapprox(mth[a], mh[a], atol=1e-4) "mh[a] mth[a],  $(mh[a]) $(mth[a])"
        gg[a] = 1/den1^2 * jd0 / j00
        g̃g̃[a] = 1/den2^2 * (j02 / j00 + j0d / j00)
        # # a==1 &&(println("g̃g̃[a]=$(g̃g̃[a]) j02=$j02 j0d=$j0d j00=$j00"))

        Oh[1] += 1/den1^2 * jd0 / j00
        Õh[2] += 1/den2^2 * (j02 / j00 + j0d / j00)

        @assert isfinite(mh[a])
        @assert isfinite(ρh[a])
    end
    q̂0 = 0.;q̂1=0.;q̂t=0.; ŝ=0.;ŝt=0.; m̂=0.;m̂t=0.
    for a=1:M
        m̂ += mh[a]
        m̂t += mth[a]
        q̂0 += mh[a]*mh[a]
        q̂1 += ρh[a]
        q̂t += mth[a]^2
        ŝ += ρth[a]
        ŝt += mh[a]*mth[a]
    end
    q̂1<q̂0 && (q̂1=q̂0; print("!h"))
    @assert q̂1>=q̂0 "q1>=q0 q0=$q0 q1=$q1"
    # println("q̂0=$(q̂0/N) q̂1=$(q̂1/N) q̂t=$(q̂t/N) ŝ=$(ŝ/N) ŝt=$(ŝt/N)")

    Oh[:] += [y*(q̂1-q̂0), ŝ-ŝt] # some terms added before
    Õh[:] += [y*(ŝ-ŝt), -q̂t]
    Δ = 0.
    # variables update
    for i=1:N
        Mtot = 0.
        M̃tot = 0.
        for a=1:M
            Mtot += ξ[i, a]* mh[a]
            M̃tot += ξ[i, a]* mth[a]
        end
        Mtot += (-m[i]*Oh[1] - mt[i]*Oh[2])
        M̃tot += (-m[i]*Õh[1] - mt[i]*Õh[2])
        coeffs = (Mtot, √(q̂1-q̂0), M̃tot)
        γeff = ŝ - ŝt + γ
        k00 = K00(coeffs..., γeff, y)
        k10 = K10(coeffs..., γeff, y)
        k20 = K20(coeffs..., γeff, y)
        k01 = K01(coeffs..., γeff, y)
        k11 = K11(coeffs..., γeff, y)
        # println(m[i]," ", k0, " ", k1, " ", k2)

        # @assert isapprox(mt[i], m[i], atol=1e-4) "i=$i m[i] mt[i],  $(m[i]) $(mt[i])"
        # @assert isapprox(mt[i], tanh(M̃tot), atol=1e-6) "i=$i m[i] mt[i],  $(m[i]) $(mt[i])"

        # i==1 &&(println("M̃tot=$M̃tot Õh=$Õh mt[i]=$(mt[i]) q̂t=$(q̂t)"))
        tf.ϕ += log(k00)
        ϕ0 = log(k00)
        # tf.Σint += KY(coeffs..., γeff, y)/k00

        if computetf
            # println("computetf")
            for a=1:M
                rh = [y*(ρh[a]-mh[a]^2)+gg[a], ρth[a]-mh[a]*mth[a]]
                r̃h = [y*(ρth[a]-mh[a]*mth[a]), g̃g̃[a] - mth[a]^2]

                ϵ = ξ[i, a]* mh[a] - m[i]*rh[1] - mt[i]*rh[2]
                ϵt = ξ[i, a]* mth[a] - m[i]*r̃h[1] - mt[i]*r̃h[2]
                ϵs = ρth[a] - mth[a]*mh[a]
                ϵq = ρh[a] - mh[a]^2
                Mcav = Mtot - ϵ
                M̃cav = M̃tot - ϵt
                γcav = γeff - ϵs
                Qcav = q̂1-q̂0 - ϵq
                Qcav < 0 && (Qcav=0.)
                coeffs = (Mcav, √Qcav, M̃cav, γcav)
                k00cav = K00(coeffs..., y)
                # println("computetf $i $a")

                # δ1 = (ϵt*mt[i]  + 0.5*ϵt^2*(1-mt[i]^2))
                #     #  + ϵ*y*m[i] + 0.5*ϵ^2*(y*(1-ρ[i]) + y^2*(ρ[i]-m[i]^2))
                #     #  + y*ϵt*ϵ*(ρt[i] - mt[i]*m[i]))
                # δ2 =  log(k00) - ϕ0
                # @assert isapprox(δ1, δ2, atol=1e-4) "$δ1 $δ2"
                Δϕ += log(k00cav) - ϕ0
                # ΔΣint += KY(coeffs..., y)/k00 - Σint0
            end
        end

        oldm = m[i]
        oldmt = mt[i]

        m[i] = (1-dump)*(k10 / k00) + dump*oldm
        ρ[i] = (1-dump)*(k20 / k00) + dump*ρ[i]
        if !fixmt
            mt[i] = (1-dump)*(k01 / k00) + dump*oldmt
        end
        ρt[i] = (1-dump)*(k11 / k00)  + dump*ρt[i]

        Δ = max(Δ, abs(m[i] - oldm))
        Δ = max(Δ, abs(mt[i] - oldmt))
    end

    # Ricalcolo i parametri d'ordine ed i termini di onsager per la non-hat
    # Dato che cmq lo faccio all'inizio della funzione questa parte è utile solo
    # nell'ultima iterazione.
    # NON LEVARE, altrimenti poi non posso calcolare bene le funzioni
    # termodinamiche
    # q0 = 0.;q1=0.;qt=0.; s=0.;st=0.; mm=0.;mmt=0.;
    # for i=1:N
    #     mm += m[i]
    #     mmt += mt[i]
    #     q0 += m[i]*m[i]
    #     q1 += ρ[i]
    #     qt += mt[i]^2
    #     s += ρt[i]
    #     st += m[i]*mt[i]
    # end
    # O[:] = [y*(q1 - q0) + N - q1, s - st] #Onsager Reaction on m, coefficients of mh and mth
    # Õ[:] = [y*(s - st), N - qt]           #Onsager Reaction on mt, coefficients of mh and mth


    tf.ϕ = (tf.ϕ + Δϕ)/N
    tf.ϕ += - y*γ*s + y*log(2)

    q0/=N;q1/=N;qt/=N;s/=N;st/=N; mm/=N;mmt/=N; # le hat sono già O(1) q̂0/=M;q̂1/=M;q̂t/=M;ŝ/=M;ŝt/=M
    g.op = OrderParams(mm,mmt,q0,q1,qt,s,st,m̂,m̂t,q̂0,q̂1,q̂t,ŝ,ŝt)

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
                                , altsolv::Bool=false, altconv = false, fixmt=false
                                , reinfpar::ReinfParams=ReinfParams()
                                , dump = 0.)

    ok = false
    for it=1:maxiters
        print("it=$it ...")
        Δ = oneBPiter!(g, fixmt, dump=dump)
        E = energy(g)
        @printf("r=%.3f γ=%.3f  E=%d   \tΔ=%f \n", reinfpar.r, reinfpar.γ, E, Δ)
        println(g.op)
        update_reinforcement!(reinfpar)
        g.γ=reinfpar.γ; g.y=reinfpar.r
        if altsolv && E == 0
            println("Found Solution!")
            ok = true
            break
        end
        if altconv && Δ < ϵ
            println("Converged!")
            ok = true
            break
        end
    end
    oneBPiter!(g, fixmt, dump=dump, computetf = true)
    # compute_tf!(g)
    println(g.op)
    println(g.tf)
    return ok
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

mags(g::FactorGraphTAP) = g.m
# mags_noreinf(g::FactorGraphTAP) = Float64[mag_noreinf(v) for v in g.vnodes]


function solve(; N::Int=1000, α = 0.6, seed_ξ = -1, kw...)
    if seed_ξ > 0
        srand(seed_ξ)
    end
    M = round(Int, α * N)
    ξ = rand([-1.,1.], N, M)
    σ = ones(Int, M)
    solve(ξ, σ; kw...)
end

function solve(ξ::Matrix{Float64}, σ::Vector{Int}; maxiters = 10000, ϵ = 1e-4,
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

function exclusive(f::Function, fn::AbstractString = "lock.tmp")
	run(`lockfile -1 $fn`)
	try
		f()
	finally
		run(`rm -f $fn`)
	end
end

function span(;N=2001, lstα = 0.5, lstγ = 0., n=100, y=0.,
	ϵ = 1e-4, maxiters = 100, seed = -1,
	resfile = "perc_edtap_constr.new.txt")

	global nint = n
    seed > 0 && srand(seed)
	results = Any[]
	lockfile = "reslock.tmp"

	for α in lstα
        M = round(Int, α * N)
        ξ = rand([-1.,1.], N, M)
        σ = ones(Int, M)
        # σ = rand([-1,1], M) #TODO sigma +-1 not working!!!
        g = FactorGraphTAP(ξ, σ, 0., y)
        initrand!(g)
        reinfpar = ReinfParams(y, 0., lstγ[1], 0.)
        ok = converge!(g, maxiters=maxiters, ϵ=ϵ, altsolv=false, altconv=true,reinfpar=reinfpar)
        @assert ok
        # length(lstγ) < 2 && break
        for γ in lstγ[2:end]
            g.γ=γ
            reinfpar.γ=γ
            println("\n#####  NEW ITER  ###############\n")
			ok = converge!(g, maxiters=maxiters, ϵ=ϵ, altsolv=false,
                altconv=true, fixmt=true, reinfpar=reinfpar)

            push!(results, (ok, deepcopy(g.op), deepcopy(g.tf)))
            if ok
                exclusive(lockfile) do
                    open(resfile, "a") do rf
                        print(rf, "$α $γ $y 0 ")
                        veryshortshow(rf, g.tf); print(rf, " ")
                        veryshortshow(rf, g.op); print(rf,"\n")
                    end
                end
            end
            ok || break
		end
	end
	return results
end

include("../deeplearning/deep_binary.jl")
function span_committe(;K=[301,5], α = 0.2, lstγ = 0., n=100, y=0.
        , ϵ = 1e-4, maxiters = 200, seedξ = -1, resfile = "pf_committee.new.txt"
        , ry = 0.3, ry_step = 0.01, dump=0.
        , loadξτ = "", saveξτ = "")
	global nint = n
    @assert length(K) == 3
	results = Any[]
	lockfile = "reslock.tmp"

    if loadξτ == ""
        seedξ > 0 && srand(seedξ)
        numW = length(K)==2 ? K[1]*K[2]  : sum(l->K[l]*K[l+1],1:length(K)-2)
        N = K[1]
        M = round(Int, α * numW)
        ξ = rand([-1.,1.], N, M)
        σ = ones(Int, M)
        g_deep, W_deep, E_deep, _ = DeepBinary.solve(ξ, σ; K=K,layers=[:tap,:bpex]
                           , ry=0.3, ry_step=ry_step, r_step=0., ϵ=1e-5
                           , plotinfo=0, maxiters=20000, altsolv=false, altconv=true)
        τ = [Int[convert(Int, sign(g_deep.layers[3].allmy[a][k])) for a=1:M] for k=1:K[2]]
        if saveξτ != ""
            save(saveξτ, "ξ", ξ, "τ", τ, "seedξ", seedξ
                , "ry", ry, "ry_step", ry_step)
        end
    else
        d = load(loadξτ)
        ξ=d["ξ"]; τ=d["τ"];
        N, M = size(ξ)
    end
    for k=1:1
        println("@@ Perceptron $k")
        @assert length(τ[k]) == M
        sigma = ones(Int, M)
        xi = zeros(N,M)
        for a=1:M
            for i=1:N
                xi[i,a] = ξ[i,a] * τ[k][a]
            end
        end
        # g = FactorGraphTAP(ξ, τ[k], 0., y)
        g = FactorGraphTAP(xi, sigma, 0., y)

        initrand!(g)
        reinfpar = ReinfParams(y, 0., 0., 0.)
        ok = converge!(g, maxiters=maxiters, ϵ=ϵ, altsolv=false, altconv=true,reinfpar=reinfpar, dump=dump)
        @assert ok
        for γ in lstγ
            g.γ=γ
            reinfpar.γ=γ
            println("\n#####  NEW ITER  ###############\n")
            ok = converge!(g, maxiters=maxiters, ϵ=ϵ, altsolv=false, altconv=true, fixmt=true, reinfpar=reinfpar)
            push!(results, (ok, deepcopy(g.op), deepcopy(g.tf)))
            if ok
                exclusive(lockfile) do
                    open(resfile, "a") do rf
                        print(rf, "$α $γ $y $k   ")
                        veryshortshow(rf, g.tf); print(rf, " ")
                        veryshortshow(rf, g.op); print(rf,"\n")
                    end
                end
            end
            ok || break
        end
    end
    return results
end

end #module
