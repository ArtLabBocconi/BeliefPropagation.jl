module BP
using MacroUtils

typealias Mess Float64
typealias P Ptr{Mess}
typealias VMess Vector{Mess}
typealias VPMess Vector{P}

Base.getindex(p::Ptr) = unsafe_load(p)
Base.setindex!{T}(p::Ptr{T}, x::T) = unsafe_store!(p, x)
Base.show(io::IO, p::Ptr) = show(io, p[])
Base.show(p::Ptr) = show(p[])


getref(v::Vector, i::Integer) = pointer(v, i)
Mess() = Mess(0.)

G(x) = e^(-(x^2)/2) / √(convert(typeof(x),2) * π)
H(x) = erfc(x / √convert(typeof(x),2)) / 2
#GH(x) = ifelse(x > 30.0, x+(1-2/x^2)/x, G(x) / H(x))
function GHapp(x)
    y = 1/x
    y2 = y^2
    x + y * (1 - 2y2 * (1 - 5y2 * (1 - 7.4y2)))
end
GH(x) = x > 30.0 ? GHapp(x) : G(x) / H(x)

type Fact
    m::VMess
    m̂::VPMess
    ξ::SubArray
    σ::Int
end

Fact(ξ, σ) = Fact(VMess(), VPMess(), ξ, σ)

type Var
    m̂::VMess
    m::VPMess
    #used only in BP+reinforcement
    h::Mess
end

Var() = Var(VMess(), VPMess(), Mess(0))

type FactorGraph
    N::Int
    M::Int
    ξ::Matrix{Int}
    σ::Vector{Int}
    fnodes::Vector{Fact}
    vnodes::Vector{Var}

    function FactorGraph(ξ::Matrix{Int}, σ::Vector{Int})
        N = size(ξ, 1)
        M = length(σ)
        @assert size(ξ, 2) == M
        println("# N=$N M=$M α=$(M/N)")
        fnodes = [Fact(sub(ξ, :, a), σ[a]) for a=1:M]
        vnodes = [Var() for i=1:N]

        ## Reserve memory in order to avoid invalidation of Refs
        for (a,f) in enumerate(fnodes)
            sizehint!(f.m, N)
            sizehint!(f.m̂, N)
        end
        for (i,v) in enumerate(vnodes)
            sizehint!(v.m, M)
            sizehint!(v.m̂, M)
        end

        for i=1:N, a=1:M
            f = fnodes[a]
            v = vnodes[i]

            push!(v.m̂, Mess())
            push!(f.m̂, getref(v.m̂, length(v.m̂)))

            push!(f.m, Mess())
            push!(v.m, getref(f.m, length(f.m)))
        end

        new(N, M, ξ, σ, fnodes, vnodes)
    end
end

type ReinfParams
    r::Float64
    r_step::Float64
    γ::Float64
    γ_step::Float64
    tγ::Float64
    wait_count::Int
    ReinfParams(r=0.,r_step=0.,γ=0.,γ_step=0.) = new(r, r_step, γ, γ_step, tanh(γ))
end

deg(f::Fact) = length(f.m)
deg(v::Var) = length(v.m)

function initrand!(g::FactorGraph)
    for f in g.fnodes
        f.m[:] = (2*rand(deg(f)) - 1)/2
    end
    for v in g.vnodes
        v.m̂[:] = (2*rand(deg(v)) - 1)/2
    end
end

#TODO
function update!(f::Fact)
    @extract f m m̂ σ ξ
    M = 0.
    C = float(deg(f))
    for i=1:deg(f)
        M += ξ[i]*m[i]
        C -= m[i]^2
    end
    for i=1:deg(f)
        Mcav = M - ξ[i]*m[i]
        Ccav = sqrt(C - (1-m[i]^2))
        m̂[i][] = σ*ξ[i]/Ccav * GH(-σ*Mcav / Ccav)
    end
end

function update!(v::Var, r::Float64 = 0.)
    @extract v m m̂
    Δ = 0.

    v.h = sum(m̂) + r*v.h
    ### compute cavity fields
    for a=1:deg(v)
        newm = tanh(v.h - m̂[a])
        oldm = m[a][]
        m[a][] = newm
        Δ = max(Δ, abs(newm - oldm))
    end

    Δ
end

function oneBPiter!(g::FactorGraph, r::Float64=0.)
    Δ = 0.

    for a=randperm(g.M)
        update!(g.fnodes[a])
    end

    for i=randperm(g.N)
        d = update!(g.vnodes[i], r)
        Δ = max(Δ, d)
    end

    Δ
end

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

function converge!(g::FactorGraph; maxiters::Int = 10000, ϵ::Float64=1e-5
                    , altsolv::Bool=false,altconv::Bool=true
                    , reinfpar::ReinfParams=ReinfParams())

    for it=1:maxiters
        write("it=$it ... ")
        Δ = oneBPiter!(g, reinfpar.r)
        E = energy(g)
        @printf("r=%.3f γ=%.3f  E=%d   \tΔ=%f \n", reinfpar.r, reinfpar.γ, E, Δ)
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

function energy(g::FactorGraph, W::Vector{Int})
    E = 0
    for f in g.fnodes
        E += f.σ * dot(f.ξ, W) > 0 ? 0 : 1
    end
    E
end

energy(g::FactorGraph) = energy(g, getW(mags(g)))

function mag(v::Var)
    m = tanh(v.h)
    @assert isfinite(m)
    return m
end
#
# function mag_noreinf(v::Var)
#     ispinned(v) && return float(v.pinned)
#     πp, πm = πpm(v)
#     πp /= v.ηreinfp
#     πm /= v.ηreinfm
#     m = (πp - πm) / (πm + πp)
#     # @assert isfinite(m)
#     return m
# end

mags(g::FactorGraph) = Float64[mag(v) for v in g.vnodes]
# mags_noreinf(g::FactorGraph) = Float64[mag_noreinf(v) for v in g.vnodes]


function solve(; N::Int=1000, α::Float64=0.6, seed_ξ::Int=-1, kw...)
    if seed_ξ > 0
        srand(seed_ξ)
    end
    M = round(Int, α * N)
    ξ = rand([-1,1], N, M)
    σ = ones(Int, M)
    solve(ξ, σ; kw...)
end

function solve(ξ::Matrix{Int}, σ::Vector{Int}; maxiters::Int = 10000, ϵ::Float64 = 1e-4,
                method = :reinforcement, #[:reinforcement, :decimation]
                r::Float64 = 0., r_step::Float64= 0.001,
                γ::Float64 = 0., γ_step::Float64=0.,
                altsolv::Bool = true, altconv::Bool=false,
                seed::Int = -1)

    seed > 0 && srand(seed)
    g = FactorGraph(ξ, σ)
    initrand!(g)

    # if method == :reinforcement
    reinfpar = ReinfParams(r, r_step, γ, γ_step)
    converge!(g, maxiters=maxiters, ϵ=ϵ, reinfpar=reinfpar
        , altsolv=altsolv, altconv=altconv)
    return getW(mags(g))
end

end#module
