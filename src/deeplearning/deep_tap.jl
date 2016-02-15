module Deep

using MacroUtils
using FastGaussQuadrature
using PyPlot

typealias CVec Vector{Complex128}
typealias Vec Vector{Float64}
typealias VecVec Vector{Vec}
typealias VecVecVec Vector{VecVec}

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
    K::Vector{Int} # dimension of hidden layers
    L::Int         # number of hidden layers
    ξ::Matrix{Float64}
    σ::Vector{Int}

    allm::VecVecVec
    allmy::VecVecVec
    allmh::VecVecVec

    allh::VecVecVec # for W reinforcement
    allhy::VecVecVec # for Y reinforcement

    allpu::VecVecVec # p(σ=up) from fact to y
    allpd::VecVecVec # p(σ=up) from y  to fact

    Mtot::VecVecVec
    Ctot::VecVec
    MYtot::VecVecVec
    CYtot::VecVec

    exactlayers::Vector{Int}

    function FactorGraphTAP(ξ::Matrix{Float64}, σ::Vector{Int}
                , K::Vector{Int})
        N, M = size(ξ)
        L = length(K)-1
        @assert K[1]==N
        @assert length(σ) == M
        println("# N=$N M=$M α=$(M/N)")

        # for variables W
        allm = [[zeros(K[l]) for i=1:K[l+1]] for l=1:L]
        allh = [[zeros(K[l]) for i=1:K[l+1]] for l=1:L]
        Mtot = [[zeros(K[l]) for i=1:K[l+1]] for l=1:L]
        Ctot = [zeros(K[l+1]) for l=1:L]
        # for variables Y
        allmy = [[zeros(K[l]) for a=1:M] for l=1:L]
        allhy = [[zeros(K[l]) for a=1:M] for l=1:L]
        MYtot = [[zeros(K[l]) for a=1:M] for l=1:L]
        CYtot = [zeros(M) for l=1:L]

        # for Facts
        allmh = [[zeros(M) for k=1:K[l+1]] for l=1:L]
        #
        allpu = [[zeros(M) for k=1:K[l+1]] for l=1:L]
        allpd = [[zeros(M) for k=1:K[l+1]] for l=1:L]

        new(N, M, K, L, ξ, σ
            , allm, allmy, allmh, allh, allhy, allpu,allpd
            , Mtot, Ctot, MYtot, CYtot, Vector{Int}())
    end
end

type ReinfParams
    r::Float64
    r_step::Float64
    ry::Float64
    ry_step::Float64
    wait_count::Int
    ReinfParams(r=0., r_step=0., ry=0., ry_step=0.) = new(r, r_step, ry, ry_step, 0)
end
function update_reinforcement!(reinfpar::ReinfParams)
    if reinfpar.wait_count < 10
        reinfpar.wait_count += 1
    else
        reinfpar.r = 1 - (1-reinfpar.r) * (1-reinfpar.r_step)
        reinfpar.ry = 1 - (1-reinfpar.ry) * (1-reinfpar.ry_step)

    end
end

function initrand!(g::FactorGraphTAP)
    @extract g N M L K allm allmh allmy MYtot CYtot Mtot Ctot allh allpu allpd ξ σ
    println(N, " ", L, " ", M, " ", K)
    for l=1:L
        for m in allm[l]
            m[:] = 2*rand(K[l]) - 1
        end
        for my in allmy[l]
            my[:] = 2*rand(K[l]) - 1
        end
        for mh in allmh[l]
            mh[:] = 2*rand(M) - 1
        end
        for pu in allpu[l]
            pu[:] = rand(M)
        end
        for pd in allpd[l]
            # pd[:] = ones(g.M)
            pd[:] = rand(M)
        end
    end

    # INIT IMPUT LAYER
    for a=1:M
        allmy[1][a][:] = ξ[:,a]
    end
end

expfs = Dict{Int,CVec}()
expinv0s = Dict{Int,CVec}()
expinv2ps = Dict{Int,CVec}()
expinv2ms = Dict{Int,CVec}()
expinv2Ps = Dict{Int,CVec}()
expinv2Ms = Dict{Int,CVec}()

## Utility fourier tables for the exact theta node
fexpf(N) = Complex128[exp(2π*im*p/(N+1)) for p=0:N]
fexpinv0(N) = Complex128[exp(-2π*im*p*(N-1)/2/(N+1)) for p=0:N]
fexpinv2p(N) = Complex128[(
        a =exp(-2π*im*p*(N-1)/2/(N+1));
        b = exp(-2π*im*p/(N+1));
        p==0 ? (N+1)/2 : a*b/(1-b)*(1-b^((N+1)/2)))
        for p=0:N]
fexpinv2m(N) = Complex128[(
        a =exp(-2π*im*p*(N-1)/2/(N+1));
        b = exp(2π*im*p/(N+1));
        p==0 ? (N+1)/2 : a*b/(1-b)*(1-b^((N+1)/2)))
        for p=0:N]
fexpinv2P(N) = Complex128[(
        a =exp(-2π*im*p/(N+1)*(N+1)/2);
        b = exp(-2π*im*p/(N+1));
        p==0 ? (N+1)/2 : a/(1-b)*(1-b^((N+1)/2)))
        for p=0:N]
fexpinv2M(N) = Complex128[(
        a =exp(-2π*im*p/(N+1)*(N-1)/2);
        b = exp(2π*im*p/(N+1));
        p==0 ? (N+1)/2 : a/(1-b)*(1-b^((N+1)/2)))
        for p=0:N]

function updateTopLayerExact(g::FactorGraphTAP)
    @extract g M L allm allmh allmy MYtot CYtot Mtot Ctot allh allpu allpd ξ σ
    allpu = allpu[end]
    allpd = allpd[end]
    N = g.K[end]
    λ=1
    if N == 1
        for a=1:M
            allpd[1][a] = (1+σ[a])/2
        end
        return
    end

    expf =fexpf(N)
    expinv0 = fexpinv0(N)
    expinv2p = fexpinv2p(N)
    expinv2m = fexpinv2m(N)
    expinv2P = Base.@get!(expinv2Ps, N, fexpinv2P(N))
    expinv2M = Base.@get!(expinv2Ms, N, fexpinv2M(N))

    #TODO capire perché è molto più lento facendo così
    # expf = Base.@get!(expfs, N, fexpf(N))
    # expinv0 = Base.@get!(expinv0s, N, fexpinv0(N))
    # expinv2p = Base.@get!(expinv2ps, N, fexpinv2p(N))
    # expinv2m = Base.@get!(expinv2ms, N, fexpinv2m(N))
    # expinv2P = Base.@get!(expinv2Ps, N, fexpinv2P(N))
    # expinv2M = Base.@get!(expinv2Ms, N, fexpinv2M(N))
    #
    for a=1:M
        X = ones(Complex128, N+1)
        for p=1:N+1
            for i=1:N
                X[p] *= (1-allpu[i][a]) + allpu[i][a]*expf[p]
            end
        end
        s2P = Complex128(0.)
        s2M = Complex128(0.)
        for p=1:N+1
            s2P += expinv2P[p] * X[p]
            s2M += expinv2M[p] * X[p]
        end
        # newU = (pp - pm) / (pp + pm)
        mUp = real(s2P - s2M) / real(s2P + s2M)
        @assert isfinite(mUp)

        @inbounds for i = 1:N
            pu  = allpu[i][a]
            s0 = Complex128(0.)
            s2p = Complex128(0.)
            s2m = Complex128(0.)
            for p=1:N+1
                xp = X[p] / (1-pu + pu*expf[p])
                s0 += expinv0[p] * xp
                s2p += expinv2p[p] * xp
                s2m += expinv2m[p] * xp
            end
            vH= σ[a]
            pp = (1+vH)/2; pm = 1-pp
            sr = vH * real(s0 / (pp*(s0 + 2s2p) + pm*(s0 + 2s2m)))
            sr > 1 && (sr=1.)
            sr < -1 && (sr=-1.)

            @assert isfinite(sr)
            allpd[i][a] = 0.5*(1+sr)
        end
    end
end


function updateLayerExact(g::FactorGraphTAP, l::Int)
    @extract g M L  K allm allmh allmy MYtot CYtot Mtot Ctot allh allpu allpd ξ σ
    if K[l] == 1
        for a=1:M
            allpd[1][a] = (1+σ[a])/2
        end
        return
    end
    N = K[l]

    expf =fexpf(N)
    expinv0 = fexpinv0(N)
    expinv2p = fexpinv2p(N)
    expinv2m = fexpinv2m(N)
    expinv2P = Base.@get!(expinv2Ps, N, fexpinv2P(N))
    expinv2M = Base.@get!(expinv2Ms, N, fexpinv2M(N))
    #TODO capire perché è molto più lento facendo così
    # expf = Base.@get!(expfs, N, fexpf(N))
    # expinv0 = Base.@get!(expinv0s, N, fexpinv0(N))
    # expinv2p = Base.@get!(expinv2ps, N, fexpinv2p(N))
    # expinv2m = Base.@get!(expinv2ms, N, fexpinv2m(N))
    # expinv2P = Base.@get!(expinv2Ps, N, fexpinv2P(N))
    # expinv2M = Base.@get!(expinv2Ms, N, fexpinv2M(N))
    #

    CYtot[l][:]=0
    for a=1:M
        MYtot[l][a][:]=0
    end
    for k=1:K[l+1]
        m = allm[l][k]; mh = allmh[l][k];
        Mt = Mtot[l][k]; Ct = Ctot[l];
        pd = allpd[l][k]; pu = allpu[l][k]
        Mt[:] = 0; Ct[k] = 0;
        CYt = CYtot[l]
        for a=1:M
            my = allmy[l][a]
            MYt = MYtot[l][a];
            # Mhtot = dot(sub(ξ,:,a),m)
            X = ones(Complex128, N+1)
            for p=1:N+1
                for i=1:N
                    pup = (1+my[i]*m[i])/2
                    X[p] *= (1-pup) + pup*expf[p]
                end
            end

            s2P = Complex128(0.)
            s2M = Complex128(0.)
            for p=1:N+1
                s2P += expinv2P[p] * X[p]
                s2M += expinv2M[p] * X[p]
            end
            # newU = (pp - pm) / (pp + pm)
            mUp = real(s2P - s2M) / real(s2P + s2M)
            @assert isfinite(mUp)
            pu[a] = (1+mUp)/2


            @inbounds for i = 1:N
                pup = (1+my[i]*m[i])/2
                s0 = Complex128(0.)
                s2p = Complex128(0.)
                s2m = Complex128(0.)
                for p=1:N+1
                    xp = X[p] / (1-pup + pup*expf[p])
                    s0 += expinv0[p] * xp
                    s2p += expinv2p[p] * xp
                    s2m += expinv2m[p] * xp
                end
                vH = 2pd[a]-1
                pp = (1+vH)/2; pm = 1-pp
                sr = vH * real(s0 / (pp*(s0 + 2s2p) + pm*(s0 + 2s2m)))
                sr > 1 && (sr=1.)
                sr < -1 && (sr=-1.)
                @assert isfinite(sr)
                MYt[i] +=  m[i] * sr
                Mt[i] +=  my[i] * sr
            end
        end
    end
end

function updateLayerApprox(g::FactorGraphTAP, l::Int)
    @extract g M L  K allm allmh allmy MYtot CYtot Mtot Ctot allh allpu allpd ξ σ
    CYtot[l][:]=0
    for a=1:M
        MYtot[l][a][:]=0
    end
    for k=1:K[l+1]
        m = allm[l][k]; mh = allmh[l][k];
        Mt = Mtot[l][k]; Ct = Ctot[l];
        pd = allpd[l][k]; pu = allpu[l][k]
        Mt[:] = 0; Ct[k] = 0;
        CYt = CYtot[l]
        for a=1:M
            my = allmy[l][a]
            MYt = MYtot[l][a];
            # Mhtot = dot(sub(ξ,:,a),m)
            Mhtot = 0.
            Chtot = 0.
            for i=1:K[l]
                Mhtot += my[i]*m[i]
                Chtot += 1 - my[i]^2*m[i]^2
            end
            Mhtot += -mh[a]*Chtot
            if Chtot == 0
                print("x2")
                # println("Chtot=$Chtot l=$l k=$k a=$a m=$m my=$my")
                Chtot = 1e-8
            end
            Hp = H(-Mhtot / √Chtot); Hm = 1-Hp
            Gp = G(-Mhtot / √Chtot); Gm = Gp
            @assert isfinite(pd[a]) "$(pd)"
            if pd[a]*Hp + (1-pd[a])*Hm <= 0
                println("pd[a]*Hp + (1-pd[a])*Hm <= 0")
                pd[a] = pd[a]<= 0. ? 1e-8 : 1-1e-8
            end
             mh[a] = 1/√Chtot*(pd[a]*Gp-(1-pd[a])*Gm) / (pd[a]*Hp+(1-pd[a])*Hm)

            # m̂[a] = 1 / √Ctot * (pd[a]*Gp - (1-pd[a])*Gm) / (pd[a]*Hp + (1-pd[a])*Hm)
            if !isfinite(mh[a])
                println(Chtot)
                println(Gp)
                println(pd[a])
            end
            @assert isfinite(mh[a])

            c = mh[a] * (Mhtot / Chtot + mh[a])
            Ct[k] += c
            if l > 1
                CYt[a] += c
                for i=1:K[l]
                    MYt[i] += m[i] * mh[a]
                end
            end
            for i=1:K[l]
                Mt[i] += my[i] * mh[a]
            end
            pu[a] = Hp
            @assert isfinite(pu[a])
            pu[a] < 0 && (pu[a]=1e-8)
            pu[a] > 1 && (pu[a]=1-1e-8)
        end
    end
end

function updateTopLayerApprox(g::FactorGraphTAP)
    @extract g N M L allm allmh allmy MYtot CYtot Mtot Ctot allh allpu allpd ξ σ
    pu = allpu[end]
    pd = allpd[end]
    K = g.K[end]
    if K == 1
        for a=1:M
            pd[1][a] = (1+σ[a])/2
        end
        return
    end

    for a=1:M
        Mhtot = 0.
        Chtot = 0.
        for i=1:K
            my = 2pu[i][a]-1
            Mhtot += my
            Chtot += 1 - my^2
        end
        # @assert Chtot > 0 "Chtot > 0 $Chtot"
        if Chtot ==0
            print("x1")
            Chtot = 1e-8
        end
        for i=1:K
            my = 2pu[i][a]-1
            Mcav = Mhtot - my
            Ccav = Chtot- (1 - my^2)

            @assert isfinite(Mcav)
            @assert isfinite(Ccav)
            # pp = H(-σ[a]*(Mcav+1) / √Ccav)
            # pm = H(-σ[a]*(Mcav-1) / √Ccav)
            # pd[i][a] = pp/(pp+pm)
            x = σ[a]*Mcav/ √Chtot
            if abs(x) < 1e5
                m = 1/ √Ccav*GH(-x)
            else
                m = x > 0 ? 1 - 1e-15 : -1 + 1e-15
            end
            pd[i][a] = (1+m)/2
            @assert isfinite(pd[i][a])  "$m $(-σ[a]*Mcav/ √Chtot) $Mcav $Mhtot $Chtot"
            # @assert isfinite(pd[i][a]) "isfinite(pd[i][a]) $pp $pm $Mcav $Ccav"
            pd[i][a] < 0 && (pd[i][a]=1e-8)
            pd[i][a] > 1 && (pd[i][a]=1-1e-8)
        end

    end

end

function oneBPiter!(g::FactorGraphTAP, r::Float64=0., ry::Float64=0.)
    @extract g N M L K allm allmh allmy MYtot CYtot Mtot Ctot allh  allhy allpu allpd ξ σ
    Δ = 0.
    for l=1:L
        ## factors update ################
        if l in g.exactlayers
            updateLayerExact(g, l)
        else
            updateLayerApprox(g, l)
        end
        ######################################

        ## variables W update ################
        for k=1:K[l+1]
            m=allm[l][k];
            Mt=Mtot[l][k]; Ct = Ctot[l];
            h=allh[l][k]
            @inbounds for i=1:K[l]
                h[i] = Mt[i] + m[i] * Ct[k] + r*h[i]
                oldm = m[i]
                m[i] = tanh(h[i])
                Δ = max(Δ, abs(m[i] - oldm))
            end
        end
        #########################################

        ## variables Y update ################
        if l>1 ## SKIP IMPUT LAYER
            for a=1:M
                MYt=MYtot[l][a]; CYt = CYtot[l][a]; my=allmy[l][a]; hy=allhy[l][a]
                @inbounds for i=1:K[l]
                    hy[i] = MYt[i] + my[i] * CYt + ry* hy[i]
                    @assert isfinite(hy[i])
                    allpd[l-1][i][a] = (1+tanh(hy[i]))/2
                    @assert isfinite(allpd[l-1][i][a]) "isfinite(allpd[l-1][i][a]) $(MYt[i]) $(my[i] * CYt) $(hy[i])"

                    pu = allpu[l-1][i][a];
                    if pu > 1-1e-10 || pu < 1e-10
                        hy[i] = pu > 0.5 ? 100 : -100
                        my[i] = 2pu-1
                    else
                    # @assert (pu > 0 && pu < 1)   "$pu $l $i $a"
                        hy[i] += atanh(2*pu -1)
                        @assert isfinite(hy[i]) "isfinite(hy[i]) pu=$pu"
                        my[i] = tanh(hy[i])
                    end
                end
            end
        end
        #########################################
    end # l=1:L-1

    updateTopLayerExact(g)
    # updateTopLayerApprox(g)

    Δ
end

getW(mags::VecVecVec) = [[Float64[1-2signbit(m) for m in magk]
                        for magk in magsl] for magsl in mags]

function print_overlaps{T}(W::Vector{Vector{Vector{T}}}; meanvar = true)
    K = map(length, W)
    L = length(K)
    N = length(W[1][1])
    allq = [ Vec() for l=1:L]
    clf()
    for l=1:L
        q = allq[l]
        norm = l > 1? K[l-1] : N
        for k=1:K[l]
            for p=k+1:K[l]
                push!(q, dot(W[l][k],W[l][p])/norm)
            end
        end
        subplot(L*100 + 1*10 + L-l+1)
        plt[:hist](q)
    end
end

function converge!(g::FactorGraphTAP; maxiters::Int = 10000, ϵ::Float64=1e-5
                                , altsolv::Bool=false, altconv = false
                                , reinfpar::ReinfParams=ReinfParams())

    for it=1:maxiters
        print("it=$it ... ")
        Δ = oneBPiter!(g, reinfpar.r, reinfpar.ry)
        W = getW(mags(g))
        E = energy(g, W)
        print_overlaps(W, meanvar=true)
        @printf(" r=%.3f ry=%.3f E=%d   \tΔ=%f \n", reinfpar.r, reinfpar.ry, E, Δ)
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

function energy{T}(g::FactorGraphTAP, W::Vector{Vector{Vector{T}}})
    @extract g M K L σ ξ
    E = 0
    @assert length(W) == L
    for a=1:M
        σks = ξ[:,a]
        for l=1:L
            σks = Int[ifelse(dot(σks, W[l][k]) > 0, 1, -1) for k=1:K[l+1]]
            # println(σks)
        end
        E += σ[a] * sum(σks) > 0 ? 0 : 1
    end
    E
end

energy(g::FactorGraphTAP) = energy(g, getW(mags(g)))

mags(g::FactorGraphTAP) = g.allm
# mags_noreinf(g::FactorGraphTAP) = Float64[mag_noreinf(v) for v in g.vnodes]


function solve(; K::Vector{Int} = [101,3], α::Float64=0.6
            , seed_ξ::Int=-1, kw...)
    seed_ξ > 0 && srand(seed_ξ)
    num = sum(l->K[l]*K[l+1],1:length(K)-1)
    M = round(Int, α * num)
    ξ = rand([-1.,1.], K[1], M)
    σ = ones(Int, M)
    solve(ξ, σ; K=K, kw...)
end

function solve(ξ::Matrix, σ::Vector{Int}; maxiters::Int = 10000, ϵ::Float64 = 1e-4,
                K::Vector{Int} = [101, 3], exactlayers=[2],
                r::Float64 = 0., r_step::Float64= 0.001,
                ry::Float64 = 0., ry_step::Float64= 0.0,
                altsolv::Bool = true, altconv::Bool = false,
                seed::Int = -1)
    for l=1:length(K)
        @assert K[l] % 2 == 1
    end
    seed > 0 && srand(seed)
    g = FactorGraphTAP(ξ, σ, K)
    initrand!(g)
    g.exactlayers = exactlayers
    # if method == :reinforcement
    reinfpar = ReinfParams(r, r_step, ry, ry_step)
    converge!(g, maxiters=maxiters, ϵ=ϵ, reinfpar=reinfpar,
            altsolv=altsolv, altconv=altconv)
    return getW(mags(g))
end

end #module
