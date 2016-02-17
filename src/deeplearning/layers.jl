"""
Input from bottom.allpu and top.allpd
and modifies its allpu and allpd
"""

G(x) = e^(-(x^2)/2) / √(convert(typeof(x),2) * π)
H(x) = erfc(x / √convert(typeof(x),2)) / 2
#GH(x) = ifelse(x > 30.0, x+(1-2/x^2)/x, G(x) / H(x))
function GHapp(x)
    y = 1/x
    y2 = y^2
    x + y * (1 - 2y2 * (1 - 5y2 * (1 - 7.4y2)))
end
GH(x) = x > 30.0 ? GHapp(x) : G(x) / H(x)

abstract AbstractLayer

myatanh(x::Float64) = ifelse(abs(x) > 15, ifelse(x>0,50,-50), atanh(x))

###########################
#      OUTPUT LAYER
#######################################

type OutputLayer <: AbstractLayer
    l::Int
    allpd::VecVec # p(σ=up) from fact ↑ to y
end
function OutputLayer(σ::Vector)
    allpd = VecVec()
    push!(allpd, Float64[(1+σ[a])/2 for a=1:length(σ)])
    OutputLayer(-1,allpd)
end
initrand!(layer::OutputLayer) = nothing

###########################
#      INPUT LAYER
#######################################
type InputLayer <: AbstractLayer
    l::Int
    allpu::VecVec # p(σ=up) from fact ↑ to y
end
InputLayer(ξ::Matrix) = InputLayer(1,
    [Float64[(1+ξ[i,a])/2 for a=1:size(ξ,2)] for i=1:size(ξ,1)])
initrand!(layer::InputLayer) = nothing


###########################
#       BP EXACT LAYER
#######################################
type BPExactLayer <: AbstractLayer
    l::Int
    K::Int
    N::Int
    M::Int

    allm::VecVec
    allmy::VecVec
    allmh::VecVec

    allmcav::VecVecVec
    allmycav::VecVecVec
    allmhcavtoy::VecVecVec
    allmhcavtow::VecVecVec

    allh::VecVec # for W reinforcement
    allhy::VecVec # for Y reinforcement

    allpu::VecVec # p(σ=up) from fact ↑ to y
    allpd::VecVec # p(σ=up) from y  ↓ to fact

    top_allpd::VecVec
    bottom_allpu::VecVec

    expf::CVec
    expinv0::CVec
    expinv2p::CVec
    expinv2m::CVec
    expinv2P::CVec
    expinv2M::CVec

    istoplayer::Bool
end


function BPExactLayer(K::Int, N::Int, M::Int)
    # for variables W
    allm = [zeros(N) for i=1:K]
    allh = [zeros(N) for i=1:K]


    allmcav = [[zeros(N) for i=1:M] for i=1:K]
    allmycav = [[zeros(N) for i=1:K] for i=1:M]
    allmhcavtoy = [[zeros(K) for i=1:N] for i=1:M]
    allmhcavtow = [[zeros(M) for i=1:N] for i=1:K]
    # for variables Y
    allmy = [zeros(N) for a=1:M]
    allhy = [zeros(N) for a=1:M]

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

    istoplayer = K == 1

    return BPExactLayer(-1, K, N, M, allm, allmy, allmh
        , allmcav, allmycav, allmhcavtoy,allmhcavtow
        , allh, allhy, allpu,allpd
        , VecVec(), VecVec()
        , fexpf(N), fexpinv0(N), fexpinv2p(N), fexpinv2m(N), fexpinv2P(N), fexpinv2M(N)
        , istoplayer)
end


function updateFact!(layer::BPExactLayer, k::Int)
    @extract layer K N M allm allmy allmh allpu allpd
    @extract layer bottom_allpu top_allpd istoplayer
    @extract layer expf expinv0 expinv2M expinv2P expinv2m expinv2p
    @extract layer allmcav allmycav allmhcavtow allmhcavtoy

    mh = allmh[k];
    pdtop = top_allpd[k];
    pubot = bottom_allpu;
    for a=1:M
        mycav = allmycav[a][k]
        mcav = allmcav[k][a]
        mhw = allmhcavtow[k]
        mhy = allmhcavtoy[a]

        X = ones(Complex128, N+1)
        for p=1:N+1
            for i=1:N
                pup = (1+mcav[i]*mycav[i])/2
                X[p] *= (1-pup) + pup*expf[p]
            end
        end

        vH = 2pdtop[a]-1
        if !istoplayer > 0
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
        end

        for i = 1:N
            pup = (1+mcav[i]*mycav[i])/2
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

            mhw[i][a] =  atanh(mycav[i] * sr)
            mhy[i][k] =  atanh(mcav[i] * sr)
            @assert isfinite(mycav[i])
            @assert isfinite(allpd[i][a])
            @assert isfinite(sr)
        end
    end
end

function updateVarW!{L <: Union{BPExactLayer}}(layer::L, k::Int, r::Float64=0.)
    @extract layer K N M allm allmy allmh allpu allpd allh
    @extract layer bottom_allpu top_allpd istoplayer
    @extract layer expf expinv0 expinv2M expinv2P expinv2m expinv2p
    @extract layer allmcav allmycav allmhcavtow allmhcavtoy

    m = allm[k]
    h = allh[k]
    Δ = 0.
    for i=1:N
        mhw = allmhcavtow[k][i]
        mcav = allmcav[k]
        h[i] = sum(mhw) + r*h[i]
        oldm = m[i]
        m[i] = tanh(h[i])
        for a=1:M
            mcav[a][i] = tanh(h[i]-mhw[a])
        end
        Δ = max(Δ, abs(m[i] - oldm))
    end
    return Δ
end

function updateVarY!{L <: Union{BPExactLayer}}(layer::L, a::Int, ry::Float64=0.)
    @extract layer K N M allm allmy allmh allpu allpd allhy
    @extract layer bottom_allpu top_allpd istoplayer
    @extract layer expf expinv0 expinv2M expinv2P expinv2m expinv2p
    @extract layer allmcav allmycav allmhcavtow allmhcavtoy


    my = allmy[a]
    hy = allhy[a]
    for i=1:N
        mhy = allmhcavtoy[a][i]
        mycav = allmycav[a]
        pu = bottom_allpu[i][a];
        @assert pu >= 0 && pu <= 1 "$pu $i $a $(bottom_allpu[i])"

        hy[i] = sum(mhy) + ry* hy[i]
        @assert isfinite(hy[i])
        allpd[i][a] = (1+tanh(hy[i]))/2
        @assert isfinite(allpd[i][a]) "isfinite(allpd[i][a]) $(MYt[i]) $(my[i] * CYt) $(hy[i])"
        # pinned from below (e.g. from input layer)
        if pu > 1-1e-15 || pu < 1e-15
            hy[i] = pu > 0.5 ? 100 : -100
            my[i] = 2pu-1
            for k=1:K
                mycav[k][i] = 2pu-1
            end
        else
            hy[i] += atanh(2*pu -1)
            @assert isfinite(hy[i]) "isfinite(hy[i]) pu=$pu"
            my[i] = tanh(hy[i])
            for k=1:K
                mycav[k][i] = tanh(hy[i]-mhy[k])
            end
        end
    end
end

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

    istoplayer::Bool
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

    istoplayer = K == 1

    return TapExactLayer(-1, K, N, M, allm, allmy, allmh, allh, allhy, allpu,allpd
        , Mtot, Ctot, MYtot, CYtot, VecVec(), VecVec()
        , fexpf(N), fexpinv0(N), fexpinv2p(N), fexpinv2m(N), fexpinv2P(N), fexpinv2M(N)
        , istoplayer)
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
    @extract layer bottom_allpu top_allpd istoplayer
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
        if istoplayer
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
        if !istoplayer
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
        end


        for i = 1:N
            # magY = istoplayer ? 2pubot[i][a]-1 : my[i]-mh[a]*m[i]*(1-my[i]^2)
            # magW = m[i]-mh[a]*my[i]*(1-m[i]^2)
            magY = istoplayer ? 2pubot[i][a]-1 : my[i]
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
            if istoplayer
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

    istoplayer::Bool
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

    istoplayer = K == 1

    return TapLayer(-1, K, N, M, allm, allmy, allmh, allh, allhy, allpu,allpd
        , Mtot, Ctot, MYtot, CYtot, VecVec(), VecVec()
        , istoplayer)
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
        for i=1:N
            Mhtot += my[i]*m[i]
            Chtot += 1 - my[i]^2*m[i]^2
        end

        Mhtot += -mh[a]*Chtot

        if Chtot == 0
            Chtot = 1e-8
        end
        Hp = H(-Mhtot / √Chtot); Hm = 1-Hp
        Gp = G(-Mhtot / √Chtot); Gm = Gp
        # @assert isfinite(pd[a]) "$(pd)"
        if pd[a]*Hp + (1-pd[a])*Hm <= 0.
            pd[a] -= 1e-8
        end
        mh[a] = 1/√Chtot*(pd[a]*Gp - (1-pd[a])*Gm) / (pd[a]*Hp + (1-pd[a])*Hm)
        if !isfinite(mh[a])
            println(Chtot)
            println(Gp)
            println(pd[a])
        end
        # @assert isfinite(mh[a])

        c = mh[a] * (Mhtot / Chtot + mh[a])
        Ct[k] += c
        if layer.l > 2 #no need for bottom layer
            CYt[a] += c
            for i=1:N
                MYt[i] += m[i] * mh[a]
            end
        end
        for i=1:N
            Mt[i] += my[i] * mh[a]
        end

        if length(allpu) > 0
            pu = allpu[k]
            pu[a] = Hp
            # @assert isfinite(pu[a])
            pu[a] < 0 && (pu[a]=1e-8)
            pu[a] > 1 && (pu[a]=1-1e-8)
        end
    end
end

function updateVarW!{L <: Union{TapLayer,TapExactLayer}}(layer::L, k::Int, r::Float64=0.)
    @extract layer K N M allm allmy allmh allpu allpd
    @extract layer CYtot MYtot Mtot Ctot bottom_allpu allh
    Δ = 0.
    m=allm[k];
    Mt=Mtot[k]; Ct = Ctot;
    h=allh[k]
    for i=1:N
        # DEBUG
        if i==1 && k==1 && layer.l==3
            println("Mt[i]",Mt[i])
        end
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
end


function update!(layer::BPExactLayer, r::Float64, ry::Float64)
    for k=1:layer.K
        updateFact!(layer, k)
    end
    Δ = 0.
    for k=1:layer.K
        δ = updateVarW!(layer, k, r)
        Δ = max(δ, Δ)
    end

    for a=1:layer.M
        updateVarY!(layer, a, ry)
    end
    return Δ
end

function update!{L <: Union{TapLayer,TapExactLayer}}(layer::L, r::Float64, ry::Float64)
    @extract layer K N M allm allmy allmh allpu allpd CYtot MYtot Mtot Ctot istoplayer


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
    if !istoplayer
        for k=1:K
            δ = updateVarW!(layer, k, r)
            Δ = max(δ, Δ)
        end

        for a=1:M
            updateVarY!(layer, a, ry)
        end
    end
    return Δ
end

function initrand!{L <: Union{BPExactLayer, TapExactLayer,TapLayer}}(layer::L)
    @extract layer K N M allm allmy allmh allpu allpd  top_allpd
    for m in allm
        m[:] = 2*rand(N) - 1
    end
    for my in allmy
        my[:] = 2*rand(N) - 1
    end
    for mh in allmh
        mh[:] = 2*rand(M) - 1
    end
    for pu in allpu
        pu[:] = rand(M)
    end
    for pd in top_allpd
        pd[:] = rand(M)
    end

end


function Base.show{L <: Union{TapExactLayer,TapLayer}}(io::IO, layer::L)
    @extract layer K N M allm allmy allmh allpu allpd
    println(io, "m=$(allm[1])")
    println(io, "my=$(allmy[1])")
end

chain!(lay1::InputLayer, lay2::OutputLayer) = nothing
function chain!(lay1::AbstractLayer, lay2::OutputLayer)
    lay1.top_allpd = lay2.allpd
    lay2.l = lay1.l+1
end

function chain!{L <: Union{TapExactLayer,TapLayer}}(lay1::InputLayer, lay2::L)
    lay2.l = lay1.l+1
    lay2.bottom_allpu = lay1.allpu
    for a=1:lay2.M
        updateVarY!(lay2, a)
    end
end
# chain!(lay1::InputLayer, lay2::AbstractLayer) = lay2.bottom_allpu = lay1.allpu


function chain!(lay1::AbstractLayer, lay2::AbstractLayer)
    lay2.l = lay1.l+1
    lay1.top_allpd = lay2.allpd
    lay2.bottom_allpu = lay1.allpu
end
