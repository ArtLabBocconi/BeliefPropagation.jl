module Deep

using MacroUtils
using FastGaussQuadrature
using PyPlot
include("../utils/OO.jl")
import OO.@oo

typealias CVec Vector{Complex128}
typealias Vec Vector{Float64}
typealias VecVec Vector{Vec}
typealias VecVecVec Vector{VecVec}

include("layers.jl")

type FactorGraph
    K::Vector{Int} # dimension of hidden layers
    M::Int
    L::Int         # number of hidden layers
    ξ::Matrix{Float64}
    σ::Vector{Int}
    layers::Vector{AbstractLayer}

    function FactorGraph(ξ::Matrix{Float64}, σ::Vector{Int}
                , K::Vector{Int}, layertype::Vector{Symbol})
        N, M = size(ξ)
        @assert length(σ) == M
        println("# N=$N M=$M α=$(M/N)")
        @assert K[1]==N && K[end]==1
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
            elseif  layertype[l] == :bpex
                push!(layers, BPExactLayer(K[l+1], K[l], M))
                println("Created BPExactLayer")
            else
                error("Wrong Layer Symbol")
            end
        end

        push!(layers, OutputLayer(σ))
        println("Created OutputLayer")

        for l=1:L+1
            chain!(layers[l], layers[l+1])
        end

        new(K, M, L, ξ, σ, layers)
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
    for lay in layers[2:end-2]
        initrand!(lay)
    end
    for a=1:M
        layers[2].allmy[a][:] = ξ[:,a]
    end
    g.layers[end-1].allm[1][:] = 1
end

function update!(g::FactorGraph, r::Float64, ry::Float64)
    Δ = 0.
    for lay in g.layers[2:end-1]
        δ = update!(lay, r, ry)
        Δ = max(δ, Δ)
    end
    return Δ
end

getW(mags::VecVecVec) = [[Float64[1-2signbit(m) for m in magk]
                        for magk in magsl] for magsl in mags]

function print_overlaps{T}(W::Vector{Vector{Vector{T}}})
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
        xlim(-1,1)
        plt[:hist](q)
    end
end

function converge!(g::FactorGraph; maxiters::Int = 10000, ϵ::Float64=1e-5
                                , altsolv::Bool=false, altconv = false
                                , reinfpar::ReinfParams=ReinfParams())

    for it=1:maxiters
        print("it=$it ... ")
        Δ = update!(g, reinfpar.r, reinfpar.ry)

        W = getW(mags(g))
        E = energy(g, W)
        print_overlaps(W)
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

function energy{T}(g::FactorGraph, W::Vector{Vector{Vector{T}}})
    @extract g M K L σ ξ
    E = 0
    @assert length(W) == L-1
    for a=1:M
        σks = ξ[:,a]
        for l=1:L-1
            σks = Int[ifelse(dot(σks, W[l][k]) > 0, 1, -1) for k=1:K[l+1]]
        end
        E += σ[a] * sum(σks) > 0 ? 0 : 1
    end
    E
end

energy(g::FactorGraph) = energy(g, getW(mags(g)))

mags(g::FactorGraph) = [(lay.allm)::VecVec for lay in g.layers[2:end-2]]
function solve(; K::Vector{Int} = [101,3], α::Float64=0.6
            , seed_ξ::Int=-1, kw...)
    seed_ξ > 0 && srand(seed_ξ)
    num = sum(l->K[l]*K[l+1],1:length(K)-2)
    M = round(Int, α * num)
    ξ = rand([-1.,1.], K[1], M)
    σ = ones(Int, M)
    solve(ξ, σ; K=K, kw...)
end

function solve(ξ::Matrix, σ::Vector{Int}; maxiters::Int = 10000, ϵ::Float64 = 1e-4,
                K::Vector{Int} = [101, 3],layers=[:tap,:tapex,:tapex],
                r::Float64 = 0., r_step::Float64= 0.001,
                ry::Float64 = 0., ry_step::Float64= 0.0,
                altsolv::Bool = true, altconv::Bool = false,
                seed::Int = -1)
    for l=1:length(K)
        @assert K[l] % 2 == 1
    end
    seed > 0 && srand(seed)
    g = FactorGraph(ξ, σ, K, layers)
    initrand!(g)
    # if method == :reinforcement
    reinfpar = ReinfParams(r, r_step, ry, ry_step)
    converge!(g, maxiters=maxiters, ϵ=ϵ, reinfpar=reinfpar,
            altsolv=altsolv, altconv=altconv)
    return getW(mags(g))
end

end #module
