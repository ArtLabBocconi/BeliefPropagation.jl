#############################################################
#   BPRealLayer
##############################################################

mutable struct BPRealLayer <: AbstractLayer
    l::Int
    K::Int
    N::Int
    M::Int

    allm::VecVec
    allmy::VecVec
    allmh::VecVec

    allmcav::VecVecVec
    allρcav::VecVecVec

    allmycav::VecVecVec

    allmhcavtoy::VecVecVec

    allmhcavtow::VecVecVec
    allρhcavtow::VecVecVec

    allh1::VecVec # for W reinforcement
    allh2::VecVec # for W reinforcement

    allhy::VecVec # for Y reinforcement

    allpu::VecVec # p(σ=up) from fact ↑ to y
    allpd::VecVec # p(σ=up) from y  ↓ to fact

    top_allpd::VecVec
    bottom_allpu::VecVec

    top_layer::AbstractLayer
    bottom_layer::AbstractLayer

    istree::Bool
end


function BPRealLayer(K::Int, N::Int, M::Int)
    # for variables W
    allm = [zeros(N) for i=1:K]
    allh1 = [zeros(N) for i=1:K]
    allh2 = [zeros(N) for i=1:K]


    allmcav = [[zeros(N) for i=1:M] for i=1:K]
    allρcav = [[zeros(N) for i=1:M] for i=1:K]

    allmycav = [[zeros(N) for i=1:K] for i=1:M]
    allmhcavtoy = [[zeros(K) for i=1:N] for i=1:M]

    allmhcavtow = [[zeros(M) for i=1:N] for i=1:K]
    allρhcavtow = [[zeros(M) for i=1:N] for i=1:K]

    # for variables Y
    allmy = [zeros(N) for a=1:M]
    allhy = [zeros(N) for a=1:M]

    # for Facts
    allmh = [zeros(M) for k=1:K]

    allpu = [zeros(M) for k=1:K]
    allpd = [zeros(M) for k=1:N]


    return BPRealLayer(-1, K, N, M, allm, allmy, allmh
        , allmcav, allρcav, allmycav, allmhcavtoy
        , allmhcavtow, allρhcavtow
        , allh1, allh2, allhy, allpu,allpd
        , VecVec(), VecVec()
        , DummyLayer(), DummyLayer(), false)
end


function updateFact!(layer::BPRealLayer, k::Int)
    @extract layer: K N M allm allρcav allmy allmh allpu allpd
    @extract layer: bottom_allpu top_allpd
    @extract layer: allmcav allmycav allmhcavtow allρhcavtow allmhcavtoy

    mh = allmh[k];
    pd = top_allpd[k];
    for a=1:M
        my = allmycav[a][k]
        m = allmcav[k][a]
        ρ = allρcav[k][a]
        mhw = allmhcavtow[k]
        ρhw = allρhcavtow[k]

        mhy = allmhcavtoy[a]
        Mhtot = 0.
        Chtot = 0.
        if !isbottomlayer(layer)
            @assert false
            for i=1:N
                Mhtot += my[i]*m[i]
                Chtot += 1 - my[i]^2*m[i]^2
            end
        else
            for i=1:N
                Mhtot += my[i]*m[i]
                Chtot += my[i]^2*ρ[i]
            end
        end

        Chtot <= 0. && (print("*"); Chtot =1e-5)

        # println("Mhtot $a= $Mhtot pd=$(pd[a])")
        @assert isfinite(pd[a]) "$(pd)"
        # if pd[a]*Hp + (1-pd[a])*Hm <= 0.
        #     pd[a] -= 1e-8
        # end
        mh[a] = 1/√Chtot * GH(pd[a], -Mhtot / √Chtot)
        # @assert isfinite(mh[a]) "isfinite(mh[a]) pd[a]= $(pd[a]) Mhtot=$Mhtot √Chtot=$(√Chtot)"
        if !isbottomlayer(layer)
            @assert false
            # for i=1:N
            #     Mcav = Mhtot - my[i]*m[i]
            #     Ccav = sqrt(Chtot - (1-my[i]^2 * m[i]^2))
            #     # mhw[i][a] = my[i]/Ccav * GH(pd[a],-Mcav / Ccav)
            #     mhw[i][a] = myatanh(my[i]/Ccav * GH(pd[a],-Mcav / Ccav))
            # end
        else
            for i=1:N
                Mcav = Mhtot - my[i]*m[i]
                Ccav = Chtot - my[i]^2*ρ[i]^2
                Ccav <= 0. && (Ccav=1e-5)       #print("*"); )
                x = Mcav / Ccav
                gh = GH(pd[a], -x)
                @assert isfinite(gh)
                mhw[i][a] = my[i]/√Ccav * gh
                ρhw[i][a] = my[i]^2/Ccav *(x*gh + gh^2) # -∂^2 log ν(W)

                # mhw[i][a] = myatanh(my[i]/Ccav * GH(pd[a],-Mcav / Ccav))
                # mhw[i][a] = DH(pd[a], Mcav, my[i], Ccav)
                # t = DH(pd[a], Mcav, my[i], Ccav)
                # @assert abs(t-mhw[i][a]) < 1e-1 "pd=$(pd[a]) DH=$t atanh=$(mhw[i][a]) Mcav=$Mcav, my=$(my[i])"
            end
        end

        if !isbottomlayer(layer)
            @assert false
            # for i=1:N
            #     # mhy[i][k] = mhw[i][k]* m[i] / my[i]
            #     mhy[i][k] = myatanh(m[i]/Ccav * GH(pd[a],-Mcav / Ccav))
            # end
        end

        allpu[k][a] = atanh2Hm1(-Mhtot / √Chtot)
    end
end

function updateVarW!(layer::BPRealLayer, k::Int, r::Float64=0.)
    @extract layer K N M allm allmy allmh allpu allpd allh1 allh2
    @extract layer bottom_allpu top_allpd
    @extract layer allmcav allρcav allmycav allmhcavtow allρhcavtow allmhcavtoy

    m = allm[k]
    h1 = allh1[k]
    h2 = allh2[k]
    Δ = 0.
    # println("ranfeW $k, ",rangeW(layer,k))
    # println("m $k, " ,m)

    for i in rangeW(layer,k)
        mhw = allmhcavtow[k][i]
        ρhw = allρhcavtow[k][i]
        mcav = allmcav[k]
        ρcav = allρcav[k]
        h1[i] = sum(mhw) + r*h1[i]
        h2[i] = 1. + sum(ρhw) + r*h2[i]
        # @assert h2[i] > 0.
        h2[i]<0 && (print("![]!"); continue)
        # h2[i]<0 && (print("![]!"); h2[i] = 1e-8)
        oldm = m[i]
        m[i] = h1[i] / h2[i]
        for a=1:M
            h1cav =  h1[i] - mhw[a]
            h2cav =  h2[i] - ρhw[a]
            # h2cav<0 && (print("!"); h2cav = 1e-5)
            h2cav<0 && (h2cav = 1e-8)

            mcav[a][i] = h1cav / h2cav
            ρcav[a][i] = 1 / h2cav
        end
        Δ = max(Δ, abs(m[i] - oldm))
    end
    return Δ
end

function initYBottom!(layer::BPRealLayer, a::Int)
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

function updateVarY!(layer::BPRealLayer, a::Int, ry::Float64=0.)
    @extract layer K N M allm allmy allmh allpu allpd allhy
    @extract layer bottom_allpu top_allpd
    @extract layer allmcav allmycav allmhcavtow allmhcavtoy

    @assert !isbottomlayer(layer)

    # @assert false
    my = allmy[a]
    hy = allhy[a]
    for i=1:N
        mhy = allmhcavtoy[a][i]
        mycav = allmycav[a]

        hy[i] = sum(mhy) + ry* hy[i]
        @assert isfinite(hy[i]) "isfinite(hy[i]) mhy=$mhy"
        allpd[i][a] = hy[i]

        pu = bottom_allpu[i][a];
        hy[i] += pu
        my[i] = tanh(hy[i])
        @assert isfinite(my[i]) "isfinite(my[i]) pu=$pu"
        for k=1:K
            mycav[k][i] = tanh(hy[i]-mhy[k])
        end
    end
end

function update!(layer::BPRealLayer, r::Float64, ry::Float64)
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


function initrand!(layer::BPRealLayer)
    @extract layer: K N M allm allmy allmh allpu allpd  top_allpd
    @extract layer: allmcav allρcav allmycav allmhcavtow allρhcavtow allmhcavtoy

    for m in allm
        m .= 2*rand(N) .- 1
    end
    for my in allmy
        my .= 2*rand(N) .- 1
    end
    for mh in allmh
        mh .= 2*rand(M) .- 1
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
        allρcav[k][a][i] = 1e-1

        allmycav[a][k][i] = allmy[a][i]
        allmhcavtow[k][i][a] = allmh[k][a]*allmy[a][i]
        allρhcavtow[k][i][a] = 1e-1
        allmhcavtoy[a][i][k] = allmh[k][a]*allm[k][i]
    end

end

function fixW!(layer::BPRealLayer, w=1.)
    @extract layer K N M allm allmy allmh allpu allpd top_allpd
    @extract layer allmcav allmycav allmhcavtow allmhcavtoy

    for k=1:K,i=1:N
        allm[k][i] = w
    end
    for k=1:K, a=1:M, i=1:N
        allmcav[k][a][i] = allm[k][i]
    end
end

function fixY!(layer::BPRealLayer, ξ::Matrix)
    @extract layer K N M allm allmy allmh allpu allpd top_allpd
    @extract layer allmcav allmycav allmhcavtow allmhcavtoy

    for a=1:M,i=1:N
        allmy[a][i] = ξ[i,a]
    end
    for a=1:M, k=1:K, i=1:N
        allmycav[a][k][i] = allmy[a][i]
    end
end

function rangeW(layer::BPRealLayer, k)
    @extract layer: N K istree
    if istree
        n = div(N,K)
        return (k-1)*n+1:k*n
    else
        return 1:N
    end
end

function maketree!(layer::BPRealLayer)
    @extract layer: K N M allm allmy allmh allpu allpd top_allpd
    @extract layer: allmcav allρcav allmycav allmhcavtow allρhcavtow allmhcavtoy

    @assert N % K == 0
    layer.istree = true
    for k=1:K
        for i=1:N
            (i in rangeW(layer, k)) && continue
            allm[k][i] = 0
        end
    end
    for k=1:K, a=1:M
        for i=1:N
            (i in rangeW(layer, k)) && continue
            allmcav[k][a][i] = 0
            allρcav[k][a][i] = 0
        end
    end
end
