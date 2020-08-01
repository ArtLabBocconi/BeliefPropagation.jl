#TODO Layer not working


mutable struct BPExactLayer <: AbstractLayer
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

    top_layer::AbstractLayer
    bottom_layer::AbstractLayer
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


    return BPExactLayer(-1, K, N, M, allm, allmy, allmh
        , allmcav, allmycav, allmhcavtoy,allmhcavtow
        , allh, allhy, allpu,allpd
        , VecVec(), VecVec()
        , fexpf(N), fexpinv0(N), fexpinv2p(N), fexpinv2m(N), fexpinv2P(N), fexpinv2M(N)
        , DummyLayer(), DummyLayer())
end


function updateFact!(layer::BPExactLayer, k::Int)
    @extract layer K N M allm allmy allmh allpu allpd
    @extract layer bottom_allpu top_allpd
    @extract layer expf expinv0 expinv2M expinv2P expinv2m expinv2p
    @extract layer allmcav allmycav allmhcavtow allmhcavtoy

    mh = allmh[k];
    pdtop = top_allpd[k];
    for a=1:M
        mycav = allmycav[a][k]
        mcav = allmcav[k][a]
        mhw = allmhcavtow[k]
        mhy = allmhcavtoy[a]

        X = ones(Complex{Float64}, N+1)
        for p=1:N+1
            for i=1:N
                pup = (1+mcav[i]*mycav[i])/2
                X[p] *= (1-pup) + pup*expf[p]
            end
        end

        vH = tanh(pdtop[a])
        if !istoplayer(layer)
            s2P = Complex{Float64}(0.)
            s2M = Complex{Float64}(0.)
            for p=1:N+1
                s2P += expinv2P[p] * X[p]
                s2M += expinv2M[p] * X[p]
            end
            s2PP = abs(real(s2P)) / (abs(real(s2P)) + abs(real(s2M)))
            s2MM = abs(real(s2M)) / (abs(real(s2P)) + abs(real(s2M)))
            allpu[k][a] = myatanh(s2PP, s2MM)
            mh[a] = ((1+vH)*s2PP - (1-vH)*s2MM) / ((1+vH)*s2PP + (1-vH)*s2MM)
        end

        for i = 1:N
            pup = (1+mcav[i]*mycav[i])/2
            s0 = Complex{Float64}(0.)
            s2p = Complex{Float64}(0.)
            s2m = Complex{Float64}(0.)
            for p=1:N+1
                xp = X[p] / (1-pup + pup*expf[p])
                s0 += expinv0[p] * xp
                s2p += expinv2p[p] * xp
                s2m += expinv2m[p] * xp
            end
            pp = (1+vH)/2; pm = 1-pp
            sr = vH * real(s0 / (pp*(s0 + 2s2p) + pm*(s0 + 2s2m)))
            sr > 1 && (sr=1 - 1e-10) #print("!")
            sr < -1 && (sr=-1 + 1e-10) #print("!")
            if !istoplayer(layer) || isonlylayer(layer)
                mhw[i][a] =  myatanh(mycav[i] * sr)
                !isfinite(mhw[i][a]) && (mhw[i][a] = sign(mhw[i][a])*20) #print("!")
                @assert isfinite(mhw[i][a]) "mhw[i][a]=$(mhw[i][a]) $(mycav[i]) $sr"
            end
            if !isbottomlayer(layer)
                mhy[i][k] =  myatanh(mcav[i] * sr)
                !isfinite(mhy[i][k]) && (mhy[i][k] = sign(mhy[i][k])*20) #print("!")
                @assert isfinite(mhy[i][k]) "mhy[i][k]=$(mhy[i][k]) $(mcav) $sr"
            end
            @assert isfinite(mycav[i])
            @assert isfinite(allpd[i][a])
            @assert isfinite(sr)
        end
    end
end

#############################################################
#   BPLayer
##############################################################

mutable struct BPLayer <: AbstractLayer
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

    top_layer::AbstractLayer
    bottom_layer::AbstractLayer
end


function BPLayer(K::Int, N::Int, M::Int)
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


    return BPLayer(-1, K, N, M, allm, allmy, allmh
        , allmcav, allmycav, allmhcavtoy,allmhcavtow
        , allh, allhy, allpu,allpd
        , VecVec(), VecVec()
        , DummyLayer(), DummyLayer())
end


function updateFact!(layer::BPLayer, k::Int)
    @extract layer K N M allm allmy allmh allpu allpd
    @extract layer bottom_allpu top_allpd
    @extract layer allmcav allmycav allmhcavtow allmhcavtoy

    mh = allmh[k];
    pd = top_allpd[k];
    for a=1:M
        my = allmycav[a][k]
        m = allmcav[k][a]
        mhw = allmhcavtow[k]
        mhy = allmhcavtoy[a]
        Mhtot = 0.
        Chtot = 0.
        if !isbottomlayer(layer)
            for i=1:N
                Mhtot += my[i]*m[i]
                Chtot += 1 - my[i]^2*m[i]^2
            end
        else
            for i=1:N
                Mhtot += my[i]*m[i]
                Chtot += my[i]^2*(1 - m[i]^2)
            end
        end

        Chtot == 0 &&  (Chtot = 1e-8); # print("!")

        # println("Mhtot $a= $Mhtot pd=$(pd[a])")
        # @assert isfinite(pd[a]) "$(pd)"
        # if pd[a]*Hp + (1-pd[a])*Hm <= 0.
        #     pd[a] -= 1e-8
        # end
        mh[a] = 1/√Chtot * GH(pd[a], -Mhtot / √Chtot)
        @assert isfinite(mh[a])
        if !isbottomlayer(layer)
            for i=1:N
                Mcav = Mhtot - my[i]*m[i]
                Ccav = sqrt(Chtot - (1-my[i]^2 * m[i]^2))
                Ccav == 0 &&  (Ccav = 1e-8); # print("!")
                # mhw[i][a] = my[i]/Ccav * GH(pd[a],-Mcav / Ccav)
                gh = GH(pd[a],-Mcav / Ccav)
                # mhw[i][a] = myatanh(my[i]/Ccav * gh)
                # mhy[i][k] = myatanh(m[i]/Ccav * gh)
                @assert isfinite(gh)
                mhw[i][a] = my[i]/Ccav * gh
                mhy[i][k] = m[i]/Ccav * gh
                @assert isfinite(mhy[i][k]) "isfinite(mhy[i][k]) gh=$gh Ccav=$Ccav"
            end
        else
            for i=1:N
                Mcav = Mhtot - my[i]*m[i]
                Ccav = sqrt(Chtot - my[i]^2*(1-m[i]^2))
                mhw[i][a] = my[i]/Ccav * GH(pd[a],-Mcav / Ccav)
                # mhw[i][a] = myatanh(my[i]/Ccav * GH(pd[a],-Mcav / Ccav))
                # mhw[i][a] = DH(pd[a], Mcav, my[i], Ccav)
                # t = DH(pd[a], Mcav, my[i], Ccav)
                # @assert abs(t-mhw[i][a]) < 1e-1 "pd=$(pd[a]) DH=$t atanh=$(mhw[i][a]) Mcav=$Mcav, my=$(my[i])"
            end
        end

        allpu[k][a] = atanh2Hm1(-Mhtot / √Chtot)
    end
end

function updateVarW!(layer::L, k::Int, r::Float64=0.) where {L <: Union{BPLayer, BPExactLayer}}
    @extract layer K N M allm allmy allmh allpu allpd allh
    @extract layer bottom_allpu top_allpd
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

function updateVarY!(layer::L, a::Int, ry::Float64=0.) where {L <: Union{BPLayer, BPExactLayer}}
    @extract layer K N M allm allmy allmh allpu allpd allhy
    @extract layer bottom_allpu top_allpd
    @extract layer allmcav allmycav allmhcavtow allmhcavtoy

    @assert !isbottomlayer(layer)

    my = allmy[a]
    hy = allhy[a]
    for i=1:N
        mhy = allmhcavtoy[a][i]
        mycav = allmycav[a]
        hy[i] = sum(mhy) + ry* hy[i]
        # @assert isfinite(hy[i]) "isfinite(hy[i]) mhy=$mhy"
        allpd[i][a] = hy[i]
        # (allpd[i][a] < 0.) && (print("!y");allpd[i][a] = 1e-10)
        # (allpd[i][a] > 1.) && (print("!y");allpd[i][a] = 1-1e-10)
        # @assert isfinite(allpd[i][a]) "isfinite(allpd[i][a]) $(MYt[i]) $(my[i] * CYt) $(hy[i])"
        # pinned from below (e.g. from input layer)
        # if pu > 1-1e-10 || pu < 1e-10
        #     hy[i] = pu > 0.5 ? 100 : -100
        #     my[i] = 2pu-1
        #     for k=1:K
        #         mycav[k][i] = 2pu-1
        #     end
        # else
        pu = bottom_allpu[i][a];
        hy[i] += pu
        my[i] = tanh(hy[i])
        @assert isfinite(my[i]) "isfinite(my[i]) pu=$pu mhy=$mhy $(typeof(layer)))"
        for k=1:K
            mycav[k][i] = tanh(hy[i]-mhy[k])
        end
    end
end

function initYBottom!(layer::L, a::Int, ry::Float64=0.) where {L <: Union{BPLayer, BPExactLayer}}
    @extract layer K N M allm allmy allmh allpu allpd allhy
    @extract layer bottom_allpu top_allpd
    @extract layer allmcav allmycav allmhcavtow allmhcavtoy

    @assert isbottomlayer(layer)

    my = allmy[a]
    ξ = layer.bottom_layer.ξ
    for i=1:N
        my[i] = ξ[i, a]
        mycav = allmycav[a]
        for k=1:K
            mycav[k][i] = ξ[i, a]
        end
    end
end



function randupdate!(layer::L, r::Float64, ry::Float64) where {L <: Union{BPLayer, BPExactLayer}}
    @extract layer K N M allm allmy allmh allpu allpd allhy
    @extract layer bottom_allpu top_allpd
    @extract layer allmcav allmycav allmhcavtow allmhcavtoy


    updateFact!(layer, rand(1:K))
    # println("mhcavw=$(allmhcavtow[1][1])")
    Δ = 0.
    if !istoplayer(layer) || isonlylayer(layer)
        # println("Updating W")
        δ = updateVarW!(layer, rand(1:K), r)
        Δ = max(δ, Δ)
    end
    if !isbottomlayer(layer)
        updateVarY!(layer, rand(1:M), ry)
    end
    return Δ
end

function update!(layer::L, r::Float64, ry::Float64) where {L <: Union{BPLayer, BPExactLayer}}
    @extract layer K N M allm allmy allmh allpu allpd allhy
    @extract layer bottom_allpu top_allpd
    @extract layer allmcav allmycav allmhcavtow allmhcavtoy

    # println("m=$(allm[1])")
    # println("mcav=$(allmcav[1][1])")
    for k=1:K
        updateFact!(layer, k)
    end
    # println("mhcavw=$(allmhcavtow[1][1])")
    Δ = 0.
    if !istoplayer(layer) || isonlylayer(layer)
        # println("Updating W")
        for k=1:K
            δ = updateVarW!(layer, k, r)
            Δ = max(δ, Δ)
        end
    end
    if !isbottomlayer(layer)
        for a=1:M
            updateVarY!(layer, a, ry)
        end
    end
    return Δ
end


function initrand!(layer::L) where {L <: Union{BPLayer, BPExactLayer}}
    @extract layer K N M allm allmy allmh allpu allpd  top_allpd
    @extract layer allmcav allmycav allmhcavtow allmhcavtoy
    ϵ = 1e-1
    for m in allm
        m .= ϵ*(2*rand(N) .- 1)
    end
    for my in allmy
        my .= ϵ*(2*rand(N) .- 1)
    end
    for mh in allmh
        mh .= ϵ*(2*rand(M) .- 1)
    end
    for pu in allpu
        pu .= rand(M)
    end
    for pd in allpd
        pd .= rand(M)
    end

    # if!isbottomlayer
    for k=1:K,a=1:M,i=1:N
        allmcav[k][a][i] = allm[k][i]
        allmycav[a][k][i] = allmy[a][i]
        allmhcavtow[k][i][a] = allmh[k][a]*allmy[a][i]
        allmhcavtoy[a][i][k] = allmh[k][a]*allm[k][i]
    end

end

function fixW!(layer::L, w=1.) where {L <: Union{BPLayer, BPExactLayer}}
    @extract layer K N M allm allmy allmh allpu allpd top_allpd
    @extract layer allmcav allmycav allmhcavtow allmhcavtoy

    for k=1:K,i=1:N
        allm[k][i] = w
    end
    for k=1:K, a=1:M, i=1:N
        allmcav[k][a][i] = allm[k][i]
    end
end

function fixY!(layer::L, ξ::Matrix) where {L <: Union{BPLayer, BPExactLayer}}
    @extract layer K N M allm allmy allmh allpu allpd top_allpd
    @extract layer allmcav allmycav allmhcavtow allmhcavtoy

    for a=1:M,i=1:N
        allmy[a][i] = ξ[i,a]
    end
    for a=1:M, k=1:K, i=1:N
        allmycav[a][k][i] = allmy[a][i]
    end
end
