
###########################
#       TAP EXACT LAYER
#######################################
type TapExactLayer <: AbstractLayer
    l::Int
    K::Int
    N::Int
    M::Int

    allm::VecVec
    allmy::VecVec
    allmh::VecVec

    allh::VecVec # for W reinforcement
    allhy::VecVec # for Y reinforcement

    allpu::VecVec # p(σ=up) from fact ↑ to y
    allpd::VecVec # p(σ=up) from y  ↓ to fact

    Mtot::VecVec
    Ctot::Vec
    MYtot::VecVec
    CYtot::Vec

    top_allpd::VecVec
    bottom_allpu::VecVec

    expf::CVec
    expinv0::CVec
    expinv2p::CVec
    expinv2m::CVec
    expinv2P::CVec
    expinv2M::CVec

    top_layer::AbstractLayer
    bottom_layer::AbstractLayer
end

function TapExactLayer(K::Int, N::Int, M::Int)
    # for variables W
    allm = [zeros(N) for i=1:K]
    allh = [zeros(N) for i=1:K]
    Mtot = [zeros(N) for i=1:K]
    Ctot = zeros(K)

    # for variables Y
    allmy = [zeros(N) for a=1:M]
    allhy = [zeros(N) for a=1:M]
    MYtot = [zeros(N) for a=1:M]
    CYtot = zeros(M)

    # for Facts
    allmh = [zeros(M) for k=1:K]

    allpu = [zeros(M) for k=1:K]
    allpd = [zeros(M) for k=1:N]


    expf =fexpf(N)
    expinv0 = fexpinv0(N)
    expinv2p = fexpinv2p(N)
    expinv2m = fexpinv2m(N)
    expinv2P = fexpinv2P(N)
    expinv2M = fexpinv2M(N)

    return TapExactLayer(-1, K, N, M, allm, allmy, allmh, allh, allhy, allpu,allpd
        , Mtot, Ctot, MYtot, CYtot, VecVec(), VecVec()
        , fexpf(N), fexpinv0(N), fexpinv2p(N), fexpinv2m(N), fexpinv2P(N), fexpinv2M(N)
        , DummyLayer(), DummyLayer())
end


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



function updateFact!(layer::TapExactLayer, k::Int)
    @extract layer K N M allm allmy allmh allpu allpd
    @extract layer CYtot MYtot Mtot Ctot
    @extract layer bottom_allpu top_allpd
    @extract layer expf expinv0 expinv2M expinv2P expinv2m expinv2p
    m = allm[k]; mh = allmh[k];
    Mt = Mtot[k]; Ct = Ctot;
    pdtop = top_allpd[k];
    CYt = CYtot
    pubot = bottom_allpu;
    for a=1:M
        my = allmy[a]
        MYt = MYtot[a];
        X = ones(Complex128, N+1)
        if istoplayer(layer)
            for p=1:N+1
                for i=1:N
                    #TODO il termine di reazione si può anche omettere
                    magY = 2pubot[i][a]-1
                    magW = m[i]
                    pup = (1+magY*magW)/2
                    X[p] *= (1-pup) + pup*expf[p]
                end
            end
        else
            for p=1:N+1
                for i=1:N
                    # magY = my[i]-mh[a]*m[i]*(1-my[i]^2)
                    # magW = m[i]-mh[a]*my[i]*(1-m[i]^2)
                    # pup = (1+magY*magW)/2
                    pup = (1+my[i]*m[i])/2
                    X[p] *= (1-pup) + pup*expf[p]
                end
            end
        end

        vH = 2pdtop[a]-1
        # if !istoplayer(layer)
            s2P = Complex128(0.)
            s2M = Complex128(0.)
            for p=1:N+1
                s2P += expinv2P[p] * X[p]
                s2M += expinv2M[p] * X[p]
            end
            mUp = real(s2P - s2M) / real(s2P + s2M)
            @assert isfinite(mUp)
            allpu[k][a] = (1+mUp)/2
            mh[a] = real((1+vH)*s2P - (1-vH)*s2M) / real((1+vH)+s2P + (1-vH)*s2M)
        # end


        for i = 1:N
            # magY = istoplayer ? 2pubot[i][a]-1 : my[i]-mh[a]*m[i]*(1-my[i]^2)
            # magW = m[i]-mh[a]*my[i]*(1-m[i]^2)
            magY = istoplayer(layer) ? 2pubot[i][a]-1 : my[i]
            magW = m[i]
            pup = (1+magY*magW)/2

            s0 = Complex128(0.)
            s2p = Complex128(0.)
            s2m = Complex128(0.)
            for p=1:N+1
                xp = X[p] / (1-pup + pup*expf[p])
                s0 += expinv0[p] * xp
                s2p += expinv2p[p] * xp
                s2m += expinv2m[p] * xp
            end
            pp = (1+vH)/2; pm = 1-pp
            sr = vH * real(s0 / (pp*(s0 + 2s2p) + pm*(s0 + 2s2m)))
            sr > 1 && (sr=1.)
            sr < -1 && (sr=-1.)
            if istoplayer(layer) && !isonlylayer(layer)
                allpd[i][a] = (1 + (m[i] * sr))/2
            else
                MYt[i] +=  atanh(m[i] * sr)
                Mt[i] +=  atanh(my[i] * sr)
            end
            @assert isfinite(my[i])
            @assert isfinite(allpd[i][a])
            @assert isfinite(sr)
            if !isfinite(MYt[i])
                MYt[i] = MYt[i] > 0 ? 50 : -50
            end
        end
    end
end

###########################
#       TAP LAYER
#######################################
type TapLayer <: AbstractLayer
    l::Int

    K::Int
    N::Int
    M::Int

    allm::VecVec
    allmy::VecVec
    allmh::VecVec

    allh::VecVec # for W reinforcement
    allhy::VecVec # for Y reinforcement

    allpu::VecVec # p(σ=up) from fact ↑ to y
    allpd::VecVec # p(σ=up) from y  ↓ to fact

    Mtot::VecVec
    Ctot::Vec
    MYtot::VecVec
    CYtot::Vec

    top_allpd::VecVec
    bottom_allpu::VecVec

    top_layer::AbstractLayer
    bottom_layer::AbstractLayer
end

function TapLayer(K::Int, N::Int, M::Int)
    # for variables W
    allm = [zeros(N) for i=1:K]
    allh = [zeros(N) for i=1:K]
    Mtot = [zeros(N) for i=1:K]
    Ctot = zeros(K)

    # for variables Y
    allmy = [zeros(N) for a=1:M]
    allhy = [zeros(N) for a=1:M]
    MYtot = [zeros(N) for a=1:M]
    CYtot = zeros(M)

    # for Facts
    allmh = [zeros(M) for k=1:K]

    allpu = [zeros(M) for k=1:K]
    allpd = [zeros(M) for k=1:N]

    return TapLayer(-1, K, N, M, allm, allmy, allmh, allh, allhy, allpu,allpd
        , Mtot, Ctot, MYtot, CYtot, VecVec(), VecVec()
        , DummyLayer(), DummyLayer())
end

function updateFact!(layer::TapLayer, k::Int)
    @extract layer K N M allm allmy allmh allpu allpd CYtot MYtot Mtot Ctot bottom_allpu top_allpd

    m = allm[k]; mh = allmh[k];
    Mt = Mtot[k]; Ct = Ctot;
    pd = top_allpd[k];
    CYt = CYtot
    for a=1:M
        my = allmy[a]
        MYt = MYtot[a]
        # Mhtot = dot(sub(ξ,:,a),m)
        Mhtot = 0.
        Chtot = 0.
        #TODO controllare il termine di reazione
        if !isbottomlayer(layer)
            for i=1:N
                Mhtot += my[i]*m[i]
                Chtot += 1 - my[i]^2*m[i]^2
            end
        else
            for i=1:N
                Mhtot += my[i]*m[i]
                Chtot += my[i]^2 *(1 - m[i]^2)
            end
        end

        Mhtot += -mh[a]*Chtot

        if Chtot == 0
            Chtot = 1e-8
        end
        @assert isfinite(pd[a]) "$(pd)"
        # if pd[a]*Hp + (1-pd[a])*Hm <= 0.
        #     pd[a] -= 1e-8
        # end
        mh[a] = 1/√Chtot * GH(pd[a], -Mhtot / √Chtot)
        # mh[a] = DH(pd[a], Mhtot, √Chtot)
        if !isfinite(mh[a])
            println("mh[a]=$(mh[a]) √Chtot=$(√Chtot) Mhtot=$Mhtot")
            println("pd[a]=$(pd[a])")
            println("G=$(G(-Mhtot / √Chtot)) H=$(H(-Mhtot / √Chtot))")
            # mh[a] = 1e-10
        end
        @assert isfinite(mh[a])

        # c=0.
        c = mh[a] * (Mhtot / Chtot + mh[a])
        # c = Mhtot*GH2(pd[a], -Mhtot / √Chtot) / Chtot + mh[a]*mh[a]
        Ct[k] += c
        if !isbottomlayer(layer)
            CYt[a] += c
            for i=1:N
                MYt[i] += m[i] * mh[a]
            end
        end
        for i=1:N
            Mt[i] += my[i] * mh[a]
        end

        if !istoplayer(layer)
            pu = allpu[k]
            pu[a] = H(-Mhtot / √Chtot)
            # @assert isfinite(pu[a])
            pu[a] < 0 && (pu[a]=1e-8)
            pu[a] > 1 && (pu[a]=1-1e-8)
        end
    end
end

function updateVarW!{L <: Union{TapLayer,TapExactLayer}}(layer::L, k::Int, r::Float64=0.)
    @extract layer K N M allm allmy allmh allpu allpd l
    @extract layer CYtot MYtot Mtot Ctot bottom_allpu allh
    Δ = 0.
    m=allm[k];
    Mt=Mtot[k]; Ct = Ctot;
    h=allh[k]
    for i=1:N
        # DEBUG
        # if i==1 && k==1
        #     println("l=$l Mtot[k=1][i=1:10] = ",Mt[1:min(end,10)])
        # end
        # i==1 && println("h $(h[i]) r $r")
        h[i] = Mt[i] + m[i] * Ct[k] + r*h[i]
        oldm = m[i]
        m[i] = tanh(h[i])
        Δ = max(Δ, abs(m[i] - oldm))
    end
    return Δ
end

function updateVarY!{L <: Union{TapLayer,TapExactLayer}}(layer::L, a::Int, ry::Float64=0.)
    @extract layer K N M allm allmy allmh allpu allpd
    @extract layer allhy CYtot MYtot Mtot Ctot bottom_allpu

    MYt=MYtot[a]; CYt = CYtot[a]; my=allmy[a]; hy=allhy[a]
    if !isbottomlayer(layer)
        for i=1:N
            pu = bottom_allpu[i][a];
            @assert pu >= 0 && pu <= 1 "$pu $i $a $(bottom_allpu[i])"

            #TODO inutile calcolarli per il primo layer
            hy[i] = MYt[i] + my[i] * CYt + ry* hy[i]
            @assert isfinite(hy[i]) "MYt[i]=$(MYt[i]) my[i]=$(my[i]) CYt=$CYt hy[i]=$(hy[i])"
            allpd[i][a] = (1+tanh(hy[i]))/2
            @assert isfinite(allpd[i][a]) "isfinite(allpd[i][a]) $(MYt[i]) $(my[i] * CYt) $(hy[i])"
            # pinned from below (e.g. from input layer)
            if pu > 1-1e-10 || pu < 1e-10 # NOTE:1-e15 dà risultati peggiori
                hy[i] = pu > 0.5 ? 100 : -100
                my[i] = 2pu-1

            else
                hy[i] += atanh(2*pu -1)
                @assert isfinite(hy[i]) "isfinite(hy[i]) pu=$pu"
                my[i] = tanh(hy[i])
            end
        end
    else
        for i=1:N
            pu = bottom_allpu[i][a];
            @assert pu >= 0 && pu <= 1 "$pu $i $a $(bottom_allpu[i])"
            my[i] = 2pu-1
        end
    end
end

function update!{L <: Union{TapLayer,TapExactLayer}}(layer::L, r::Float64, ry::Float64)
    @extract layer K N M allm allmy allmh allpu allpd CYtot MYtot Mtot Ctot


    #### Reset Total Fields
    CYtot[:]=0
    for a=1:M
        MYtot[a][:]=0
    end
    for k=1:K
        Mtot[k][:] = 0; Ctot[k] = 0;
    end
    ############

    for k=1:K
        updateFact!(layer, k)
    end
    Δ = 0.
    if !istoplayer(layer) || isonlylayer(layer)
        for k=1:K
            δ = updateVarW!(layer, k, r)
            Δ = max(δ, Δ)
        end
    end

    # bypass Y if toplayer
    if !istoplayer(layer) && !isbottomlayer(layer)
        for a=1:M
            updateVarY!(layer, a, ry)
        end
    end
    return Δ
end

function initrand!{L <: Union{TapExactLayer,TapLayer}}(layer::L)
    @extract layer K N M allm allmy allmh allpu allpd  top_allpd
    ϵ = 1e-1
    for m in allm
        m[:] = (2*rand(N) - 1)*ϵ
    end
    for my in allmy
        my[:] = (2*rand(N) - 1)*ϵ
    end
    for mh in allmh
        mh[:] = (2*rand(M) - 1)*ϵ
    end
    for pu in allpu
        pu[:] = rand(M)
    end
    for pd in allpd
        pd[:] = rand(M)
    end
end

function fixW!{L <: Union{TapLayer, TapExactLayer}}(layer::L, w=1.)
    @extract layer K N M allm allmy allmh allpu allpd  top_allpd

    for k=1:K, i=1:N
        allm[k][i] = w
    end
end

function fixY!{L <: Union{TapLayer, TapExactLayer}}(layer::L, ξ::Matrix)
    @extract layer K N M allm allmy allmh allpu allpd  top_allpd

    for a=1:M, i=1:N
        allmy[a][i] = ξ[i,a]
    end
end
