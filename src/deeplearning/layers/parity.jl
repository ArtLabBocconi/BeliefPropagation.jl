
#############################################################
#   Parity
##############################################################

mutable struct ParityLayer <: AbstractLayer
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


function ParityLayer(K::Int, N::Int, M::Int)
    @assert K == 1
    @assert N == 2
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


    return ParityLayer(-1, K, N, M, allm, allmy, allmh
        , allmcav, allmycav, allmhcavtoy,allmhcavtow
        , allh, allhy, allpu,allpd
        , VecVec(), VecVec()
        , DummyLayer(), DummyLayer())
end


function updateFact!(layer::ParityLayer, k::Int)
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

        mtop = 2pd[a]-1

        mhy[1][k] = atanh(mtop*my[2])
        mhy[2][k] = atanh(mtop*my[1])
        # println(mhy)
        for i=1:N
            !isfinite(mhy[i][k]) && (mhy[i][k] = sign(mhy[i][k])*50.) # print("*y"); 
        end
    end
end

function updateVarW!(layer::ParityLayer, k::Int, r::Float64=0.)
    @extract layer K N M allm allmy allmh allpu allpd allh
    @extract layer bottom_allpu top_allpd
    @extract layer allmcav allmycav allmhcavtow allmhcavtoy

    Δ = 0.
    return Δ
end

function updateVarY!(layer::ParityLayer, a::Int, ry::Float64=0.)
    @extract layer K N M allm allmy allmh allpu allpd allhy
    @extract layer bottom_allpu top_allpd
    @extract layer allmcav allmycav allmhcavtow allmhcavtoy


    my = allmy[a]
    hy = allhy[a]
    for i=1:N
        mhy = allmhcavtoy[a][i]
        mycav = allmycav[a]
        pu = bottom_allpu[i][a];
        @assert pu >= 0 && pu <= 1 "$pu $i $a $(bottom_allpu[i])"

        hy[i] = sum(mhy) + ry* hy[i]
        @assert isfinite(hy[i]) "isfinite(hy[i]) mhy=$mhy"
        allpd[i][a] = (1+tanh(hy[i])) / 2
        (allpd[i][a] < 0.) && (print("!y");allpd[i][a] = 1e-10)
        (allpd[i][a] > 1.) && (print("!y");allpd[i][a] = 1-1e-10)
        @assert isfinite(allpd[i][a]) "isfinite(allpd[i][a]) $(MYt[i]) $(my[i] * CYt) $(hy[i])"
        # pinned from below (e.g. from input layer)
        if pu > 1-1e-10 || pu < 1e-10
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

function update!(layer::ParityLayer, r::Float64, ry::Float64)
    @extract layer K N M allm allmy allmh allpu allpd allhy
    @extract layer bottom_allpu top_allpd
    @extract layer allmcav allmycav allmhcavtow allmhcavtoy

    # println("m=$(allm[1])")
    # println("mcav=$(allmcav[1][1])")
    for k=1:K
        updateFact!(layer, k)
    end
    # println("mhcavw=$(allmhcavtow[1][1])")
    Δ = 0
    for a=1:M
        updateVarY!(layer, a, ry)
    end
    return Δ
end


function initrand!(layer::ParityLayer)
    @extract layer K N M allm allmy allmh allpu allpd  top_allpd
    @extract layer allmcav allmycav allmhcavtow allmhcavtoy

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
        allmycav[a][k][i] = allmy[a][i]
        allmhcavtow[k][i][a] = allmh[k][a]*allmy[a][i]
        allmhcavtoy[a][i][k] = allmh[k][a]*allm[k][i]
    end

end

function fixW!(layer::ParityLayer, w=1.)
    @extract layer K N M allm allmy allmh allpu allpd top_allpd
    @extract layer allmcav allmycav allmhcavtow allmhcavtoy

    for k=1:K,i=1:N
        allm[k][i] = w
    end
    for k=1:K, a=1:M, i=1:N
        allmcav[k][a][i] = allm[k][i]
    end
end

function fixY!(layer::ParityLayer, ξ::Matrix)
    @extract layer K N M allm allmy allmh allpu allpd top_allpd
    @extract layer allmcav allmycav allmhcavtow allmhcavtoy

    for a=1:M,i=1:N
        allmy[a][i] = ξ[i,a]
    end
    for a=1:M, k=1:K, i=1:N
        allmycav[a][k][i] = allmy[a][i]
    end
end
