using MacroUtils
using LightGraphs
typealias MessU Float64  # ̂ν(a→i) = P(σ_i != J_ai)
MessU()= MessU(0.)

typealias PU Ptr{MessU}
getref(v::Vector, i::Integer) = pointer(v, i)

# typealias PU Ref{MessU}
# typealias PH Ref{MessH}
# getref(v::Vector, i::Integer) = Ref(v, i)

import Base.show
Base.getindex(p::Ptr) = unsafe_load(p)
Base.setindex!{T}(p::Ptr{T}, x::T) = unsafe_store!(p, x)
Base.show(io::IO, p::Ptr) = show(io, p[])
Base.show(p::Ptr) = show(p[])

typealias VU Vector{MessU}
typealias VPU Vector{PU}

type VarIsing
    uin::Vector{MessU}
    uout::Vector{PU}
    tJ::Vector{Float64}
    Hext::Float64
end

VarIsing() = VarIsing(VU(),VPU(), Vector{Float64}(), 0.)

type FactorGraphIsing <: FactorGraph
    N::Int
    vnodes::Vector{VarIsing}
    adjlist::Vector{Vector{Int}}
end

function FactorGraphIsingRRG(N::Int, k::Int, seed_graph::Int = -1)
    g = random_regular_graph(N, k, seed_graph)
    adjlist = g.fadjlist
    assert(length(adjlist) == N)
    vnodes = [VarIsing() for i=1:N]

    for (i,v) in enumerate(vnodes)
        assert(length(adjlist[i]) == k)
        resize!(v.uin, length(adjlist[i]))
        resize!(v.uout, length(adjlist[i]))
        resize!(v.tJ, length(adjlist[i]))
    end

    for (i,v) in enumerate(vnodes)
        for (ki,j) in enumerate(adjlist[i])
            v.uin[ki] = MessU(i)
            kj = findfirst(adjlist[j], i)
            vnodes[j].uout[kj] = getref(v.uin, ki)
            v.tJ[ki] = 0.
        end
    end

    FactorGraphIsing(N, vnodes, adjlist)
end

deg(v::VarIsing) = length(v.uin)


function initrandJ!(g::FactorGraphIsing; m::Float64=0., σ::Float64=1.)
    for (i,v) in enumerate(g.vnodes)
        for (ki, j) in enumerate(g.adjlist[i])
            (i > j) && continue
            r = m + σ * (rand() - 0.5)
            g.vnodes[i].tJ[ki] = tanh(r)
            kj = findfirst(g.adjlist[j], i)
            g.vnodes[j].tJ[kj] = g.vnodes[i].tJ[ki]
        end
    end
end

function initrandH!(g::FactorGraphIsing; m::Float64=0., σ::Float64=1.)
    for v in g.vnodes
        v.Hext = m + σ * (rand() - 0.5)
    end
end

function initrandMess!(g::FactorGraphIsing; m::Float64=1., σ::Float64=1.)
    for v in g.vnodes
        for k=1:deg(v)
            v.uin[k] = m + σ * (rand() - 0.5)
        end
    end
end

#
# # function decrease_reinforcement!(g::FactorGraphKSAT, reinfp::ReinfParams)
# #     println("reinf: $(reinfp.reinf) -> $(reinfp.reinf /= 2)")
# #     println("reinf_step: $(reinfp.step) -> $(reinfp.step /= 2)")
# #     reinfp.wait_count = 0
# #
# #     for f in g.fnodes
# #         for k=1:deg(f)
# #             if !isfinite(f.πlist[k])
# #                 f.πlist[k] = rand()
# #             end
# #         end
# #     end
# #     for v in g.vnodes
# #         for k=1:degp(v)
# #             if !isfinite(v.ηlistp[k])
# #                 r = 0.5*rand()
# #                 v.ηlistp[k] = (1-2r)/(1-r)
# #             end
# #         end
# #         for k=1:degm(v)
# #             if !isfinite(v.ηlistm[k] )
# #                 r = 0.5*rand()
# #                 v.ηlistm[k] = (1-2r)/(1-r)
# #             end
# #         end
# #         v.ηreinfm = 0
# #         v.ηreinfp = 0
# #     end
# # end
#
function update!(v::VarIsing)
    @extract v uin uout tJ
    Δ = 0.
    ht = htot(v)
    for k=1:deg(v)
        hcav = ht - uin[k]
        ucav = atanh(tJ[k]*tanh(hcav))
        Δ = max(Δ, abs(ucav  - uout[k][]))
        uout[k][] = ucav
    end
    Δ
end

function oneBPiter!(g::FactorGraphIsing)
    Δ = 0.
    for i=randperm(g.N)
        d = update!(g.vnodes[i])
        Δ = max(Δ, d)
    end
    Δ
end

function converge!(g::FactorGraphIsing; maxiters::Int = 100, ϵ::Float64=1e-5)
    for it=1:maxiters
        # write("it=$it ... ")
        Δ = oneBPiter!(g)
        # @printf("Δ=%f \n", Δ)
        if Δ < ϵ
            # println("Converged!")
            break
        end
    end
end
#
# function energy(cnf::CNF, σ)
#     E = 0
#     for c in cnf.clauses
#         issatisfied = false
#         for i in c
#             if sign(i) == σ[abs(i)]
#                 issatisfied = true
#             end
#         end
#         E += issatisfied ? 0 : 1
#     end
#     E
# end
#
# function πpm(v::Var)
#     @extract v ηlistp ηlistm
#     eps = 1e-15
#     nzp = 0
#     nzm = 0
#     πp = 1.
#
#     for η in ηlistp
#     # if 1 - ηlistp[j] > eps
#         πp *= 1 - η
#     # else
#     #     # println("there")
#     #     nzp += 1
#     # end
#     end
#     πm = 1.
#     for η in ηlistm
#     # if 1 - ηlistm[j] > eps
#         πm *= 1 - η
#     # else
#     #     nzm += 1
#     # end
#     end
#     (nzp > 0 && nzm > 0) && exit("contradiction")
#     πp *= 1-v.ηreinfp
#     πm *= 1-v.ηreinfm
#
#     return πp, πm, nzp, nzm
# end
#

htot(v::VarIsing) = sum(v.uin) + v.Hext
mag(v::VarIsing) = tanh(htot(v))
mags(g::FactorGraphIsing) = Float64[mag(v) for v in g.vnodes]

function corr_conn_nn(g::FactorGraphIsing, i::Int, j::Int)
    @extract g N vnodes adjlist
    vi = vnodes[i]
    vj = vnodes[j]
    ki = findfirst(adjlist[i], j)
    kj = findfirst(adjlist[j], i)
    assert(ki >0)
    assert(kj >0)

    tJ = vi.tJ[ki]
    mij = tanh(htot(vi) - vi.uin[ki])
    mji = tanh(htot(vj) - vj.uin[kj])

    c = tJ * (1-mij^2)*(1-mji^2) / (1+tJ * mij * mji)
end

corr_disc_nn(g::FactorGraphIsing,i::Int,j::Int) = corr_conn_nn(g,i,j) + mag(g.vnodes[i])*mag(g.vnodes[j])

# function fixedpointmag(β, k)
#     u = 10.
#     tJ = tanh(β)
#     for i=1:200
#         u = atanh(tJ*(tanh((k-1)*u)))
#     end
#     tanh(k*u)
# end
#
# function crit_ferro(k)
#     res = Any[]
#     for β=0.3:0.002:0.4
#         m = fixedpointmag(β, k)
#         push!(res, (β,m))
#     end
#     return res
# end

function mainIsing(; N::Int = 1000, k::Int = 4, β::Float64 = 1., maxiters::Int = 1000, ϵ::Float64=1e-5)
    g = FactorGraphIsingRRG(N, k)
    initrandJ!(g, m=β, σ=0.)
    initrandH!(g, m=0., σ=0.)
    initrandMess!(g, m=1., σ=1.)
    converge!(g, maxiters=maxiters, ϵ=ϵ)
    return mean(mags(g))
end
