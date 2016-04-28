module DeepBinary
using HDF5
using MacroUtils
using FastGaussQuadrature
using PyPlot
include("../utils/OO.jl")
import OO.@oo

typealias CVec Vector{Complex128}
typealias IVec Vector{Int}
typealias Vec Vector{Float64}
typealias VecVec Vector{Vec}
typealias IVecVec Vector{IVec}
typealias VecVecVec Vector{VecVec}
typealias IVecVecVec Vector{IVecVec}

include("layers.jl")
include("dropout.jl")

type FactorGraph
    K::Vector{Int} # dimension of hidden layers
    M::Int
    L::Int         # number of hidden layers. L=length(layers)-2
    ξ::Matrix{Float64}
    σ::Vector{Int}
    layers::Vector{AbstractLayer}
    dropout::Dropout

    function FactorGraph(ξ::Matrix{Float64}, σ::Vector{Int}
                , K::Vector{Int}, layertype::Vector{Symbol}; β=Inf, βms = 1.,rms =1., ndrops=0)
        N, M = size(ξ)
        @assert length(σ) == M
        println("# N=$N M=$M α=$(M/N)")
        @assert K[1]==N
        L = length(K)-1
        layers = Vector{AbstractLayer}()
        push!(layers, InputLayer(ξ))
        println("Created InputLayer")
        for l=1:L
            if      layertype[l] == :tap
                push!(layers, TapLayer(K[l+1], K[l], M))
                println("Created TapLayer")
            elseif  layertype[l] == :tapex
                push!(layers, TapExactLayer(K[l+1], K[l], M))
                println("Created TapExactLayer")
            elseif  layertype[l] == :bp
                push!(layers, BPLayer(K[l+1], K[l], M))
                println("Created BPExactLayer")
            elseif  layertype[l] == :bpex
                push!(layers, BPExactLayer(K[l+1], K[l], M))
                println("Created BPExactLayer")
            elseif  layertype[l] == :ms
                push!(layers, MaxSumLayer(K[l+1], K[l], M, βms=βms, rms=rms))
                println("Created MaxSumLayer")
            elseif  layertype[l] == :parity
                @assert l == L
                push!(layers, ParityLayer(K[l+1], K[l], M))
                println("Created ParityLayer")
            elseif  layertype[l] == :bpreal
                @assert l == L
                push!(layers, BPRealLayer(K[l+1], K[l], M))
                println("Created BPRealLayer")
            else
                error("Wrong Layer Symbol")
            end
        end

        push!(layers, OutputLayer(σ,β=β))
        println("Created OutputLayer")

        for l=1:L+1
            chain!(layers[l], layers[l+1])
        end

        dropout = Dropout()
        add_rand_drops!(dropout, 3, K[2], M, ndrops)
        new(K, M, L, ξ, σ, layers, dropout)
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

function initrand!(g::FactorGraph)
    @extract g M layers K ξ
    for lay in layers[2:end-1]
        initrand!(lay)
    end
end

function fixtopbottom!(g::FactorGraph)
    @extract g M layers K ξ
    if g.L != 1
        fixW!(g.layers[end-1], 1.)
    end

    fixY!(g.layers[2], ξ)
end

function update!(g::FactorGraph, r::Float64, ry::Float64)
    Δ = 0.# Updating layer $(lay.l)")
    for l=2:g.L+1
        dropout!(g, l+1)
        δ = update!(g.layers[l], r, ry)
        Δ = max(δ, Δ)
    end
    return Δ
end

getW(mags::VecVecVec) = [[Float64[1-2signbit(m) for m in magk]
                        for magk in magsl] for magsl in mags]

function printvec(q::Vector{Float64}, head = "")
    print(head)
    if length(q) < 20
        for e in q
            @printf("%.6f ", e)
        end
    else
        @printf("mean:%.6f std:%.6f", mean(q), std(q))
    end
    println()
end
function plot_info(g::FactorGraph, info=1)
    m = mags(g)
    W = getW(m)
    K = g.K
    L = length(K)-1
    N = length(W[1][1])
    layers = g.layers[2:end-1]
    width = info
    info > 0 && clf()
    for l=1:L
        q0 = Float64[]
        for k=1:K[l+1]
            push!(q0, dot(layers[l].allm[k],layers[l].allm[k])/K[l])
        end
        qWαβ = Float64[]
        for k=1:K[l+1]
            for p=k+1:K[l+1]
                # push!(q, dot(W[l][k],W[l][p])/K[l])
                push!(qWαβ, dot(layers[l].allm[k],layers[l].allm[p]) / sqrt(q0[k]*q0[p])/K[l])
            end
        end
        printvec(q0,"layer $l q0=")
        printvec(qWαβ,"layer $l qWαβ=")

        info == 0 && continue

        subplot(L,width,width*(L-l)+1)
        title("W Overlaps Layer $l")
        xlim(-1.01,1.01)
        plt[:hist](q)
        info == 1 && continue

        subplot(L,width,width*(L-l)+2)
        title("Mags Layer $l")
        xlim(-1.01,1.01)
        plt[:hist](vcat(m[l]...))
        info == 2 && continue

        subplot(L,width,width*(L-l)+3)
        title("Fact Satisfaction Layer $l")
        xlim(-1.01,1.01)
        for k=1:K[l+1]
            pu = layers[l].allpu[k]
            pd = layers[l].top_allpd[k]
            sat = (2pu-1) .* (2pd-1)
            plt[:hist](sat)
        end
        info == 3 && continue

        subplot(L,width,width*(L-l)+4)
        title("Mag UP From Layer $l")
        xlim(-1.01,1.01)
        for k=1:K[l+1]
            pu = layers[l].allpu[k]
            plt[:hist](2pu-1)
        end
        info == 4 && continue


        subplot(L,width,width*(L-l)+5)
        title("Mag DOWN To Layer $l")
        xlim(-1.01,1.01)
        for k=1:K[l+1]
            pd = layers[l].top_allpd[k]
            plt[:hist](2pd-1)
        end
        info == 5 && continue

    end
end

function dropout!(g::FactorGraph, level::Int)
    @extract g : dropout layers
    !haskey(dropout.drops, level) && return
    pd = layers[level].allpd
    for (k, μ) in dropout.drops[level]
        pd[k][μ] = 0.5
    end
end

function converge!(g::FactorGraph; maxiters::Int = 10000, ϵ::Float64=1e-5
                                , altsolv::Bool=false, altconv = false, plotinfo=-1
                                , reinfpar::ReinfParams=ReinfParams())

    for it=1:maxiters
        Δ = update!(g, reinfpar.r, reinfpar.ry)
        E, h = energy(g, mags(g))
        @printf("it=%d \t r=%.3f ry=%.3f \t E=%d \t Δ=%f \n", it, reinfpar.r, reinfpar.ry, E, Δ)
        # println(h)
        plotinfo >=0  && plot_info(g, plotinfo)
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

function forward{T}(W::Vector{Vector{Vector{T}}}, ξ::Vector)
    L=length(W)
    K = [length(W[1][1]); map(length, W)]
    σks = deepcopy(ξ)
    stabilities = zeros(K[L])
    for l=1:L
        if l==L
            stabilities = map(w->dot(σks, w), W[L])
        end
        σks = Int[ifelse(dot(σks, W[l][k]) > 0, 1, -1) for k=1:K[l+1]]
        # for Parity Layer
        # if l!=L
        #     σks = Int[ifelse(dot(σks, W[l][k]) > 0, 1, -1) for k=1:K[l+1]]
        # else
        #     σks = Int[ifelse(prod(σks) > 0, 1, -1) for k=1:K[l+1]]
        # end
    end

    return σks, stabilities
end

function energy{T}(g::FactorGraph, W::Vector{Vector{Vector{T}}})
    @extract g M K σ ξ
    L=length(W)
    E = 0
    stabilities = zeros(M)
    for a=1:M
        σks, stab = forward(W, ξ[:,a])
        stabilities[a] = sum(stab)
        E += all((σ[a] .* σks) .> 0) ? 0 : 1
    end

    E, stabilities
end

energy(g::FactorGraph) = energy(g, getW(mags(g)))

mags(g::FactorGraph) = [(lay.allm)::VecVec for lay in g.layers[2:end-1]]

function meanoverlap(ξ::Matrix)
    N, M =size(ξ)
    q = 0.
    for a=1:M
        for b=a+1:M
            q += dot(ξ[:,a],ξ[:,b])
        end
    end
    return q / N / (0.5*M*(M-1))
end

function randTeacher(K::Vector{Int})
    L = length(K)-1
    W = Vector{Vector{Vector{Int}}}()
    for l=1:L
        push!(W, Vector{Vector{Int}}())
        for k=1:K[l+1]
            push!(W[l], rand(Int[-1,1], K[l]))
        end
    end
    if L > 1
        W[L][1][:] = 1
    end
    return W
end

function solveTS(; K::Vector{Int} = [101,3], α::Float64=0.6
            , seedξ::Int=-1
            , kw...)
    seedξ > 0 && srand(seedξ)
    numW = length(K)==2 ? K[1]*K[2]  : sum(l->K[l]*K[l+1],1:length(K)-2)
    N = K[1]
    ξ = zeros(K[1], 1)
    M = round(Int, α * numW)
    ξ = rand([-1.,1.], K[1], M)
    # ξ = (2rand(K[1], M) - 1)
    W = randTeacher(K)
    σ = Int[(res = forward(W, ξ[:,a]); res[1][1]) for a=1:M]
    # @assert size(ξ) == (N, M)
    # # println("Mean Overlap ξ $(meanoverlap(ξ))")
    # g, Wnew, E, stab = solve(ξ, σ; K=K, kw...)
    #
    # reinfpar = ReinfParams(r, r_step, ry, ry_step)

    # converge!(g, maxiters=maxiters, ϵ=1e-5, reinfpar=reinfpar,
    #         altsolv=false, altconv=altconv, plotinfo=plotinfo)
    solve(ξ, σ; K=K, kw...)
end

function solve(; K::Vector{Int} = [101,3], α::Float64=0.6
            , seedξ::Int=-1
            , dξ::Vector{Float64} = Float64[], nξ::Vector{Int} = Int[]
            , kw...)

    seedξ > 0 && srand(seedξ)
    numW = length(K)==2 ? K[1]*K[2]  : sum(l->K[l]*K[l+1],1:length(K)-2)
    N = K[1]
    ξ = zeros(K[1], 1)

    if length(nξ) == 0
        M = round(Int, α * numW)
        ξ = rand([-1.,1.], K[1], M)
        # ξ = (2rand(K[1], M) - 1)
        σ = ones(Int, M)
    else
        ξ0 = rand([-1.,1.], K[1],1)
        nξ[end] = round(Int, α * numW / prod(nξ[1:end-1]))
        M = round(Int, prod(nξ))
        @assert all(dξ[1:end-1] .>= dξ[2:end])
        for l=1:length(nξ)
            nb = size(ξ0, 2)
            na = nξ[l]
            d = dξ[l]
            @assert 0 <= d <= 0.5
            pflip = 1-sqrt(1-2d)
            ξ = zeros(N, na*nb)
            for a=1:na, b=1:nb
                m = a + (b-1)*na
                for i=1:N
                    ξ[i, m] = rand() < pflip ? rand([-1.,1.]) : ξ0[i,b]
                end
            end
            ξ0 = ξ
        end
        ξ = ξ0
        σ = rand([-1,1], M)
    end
    @assert size(ξ) == (N, M)
    # println("Mean Overlap ξ $(meanoverlap(ξ))")
    solve(ξ, σ; K=K, kw...)
end

function solveMNIST(; α=0.01, K::Vector{Int} = [784,10], kw...)
    @assert K[1] == 28*28
    # @assert K[end] == 10
    N = 784; M=round(Int, α*60000)
    h5 = h5open("data/mnist/train.hdf5", "r")
    ξ0 = reshape(h5["data"][:,:,1,1:M], N, M)
    m = mean(ξ0)
    m1, m2 = minimum(ξ0), maximum(ξ0)
    Δ = max(abs(m1-m), abs(m2-m))
    ξ = zeros(N, M)
    for i=1:N, a=1:M
        ξ[i,a] = (ξ0[i,a] - m) / Δ
    end
    @assert all(-1 .<= ξ .<= 1.)
    σ = round(Int, reshape(h5["label"][:,1:M], M) + 1)
    σ = Int[σ==1 ? 1 : -1 for σ in σ]
    solve(ξ, σ; K=K, kw...)
end

function solve(ξ::Matrix, σ::Vector{Int}; maxiters::Int = 10000, ϵ::Float64 = 1e-4,
                K::Vector{Int} = [101, 3],layers=[:tap,:tapex,:tapex],
                r::Float64 = 0., r_step::Float64= 0.001,
                ry::Float64 = 0., ry_step::Float64= 0.0,
                altsolv::Bool = true, altconv::Bool = false,
                seed::Int = -1, plotinfo=-1,
                β=Inf, βms = 1., rms = 1., ndrops = 0)

    # for l=1:length(K)
    #     @assert K[l] % 2 == 1
    # end
    seed > 0 && srand(seed)
    g = FactorGraph(ξ, σ, K, layers, β=β, βms=βms, rms=rms, ndrops=ndrops)
    initrand!(g)
    fixtopbottom!(g)
    reinfpar = ReinfParams(r, r_step, ry, ry_step)

    converge!(g, maxiters=maxiters, ϵ=ϵ, reinfpar=reinfpar,
            altsolv=altsolv, altconv=altconv, plotinfo=plotinfo)

    E, stab = energy(g)
    return g, getW(mags(g)), E, stab
end

end #module
