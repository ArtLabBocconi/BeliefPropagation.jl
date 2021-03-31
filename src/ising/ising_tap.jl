mutable struct TAPGraphIsing{TJ, TH}
    N::Int
    adjlist::Vector{Vector{Int}}
    J::Vector{Vector{TH}}
    H::Vector{TJ}
    m::Vector{T}
    m_old::Vector{T}
end

mag(g::TAPGraphIsing, i::Integer) = g.m[i]
mags(g::TAPGraphIsing) = g.m
deg(g::TAPGraphIsing, i::Integer) = length(g.adjlist[i])

function TAPGraphIsing(net::Network; T=1)
    @assert has_eprop(net, "J")
    @assert has_vprop(net, "H")

    adjlist = adjacency_list(net)
    N = nv(net)
    J = [[eprop(net, e)["J"] / T for e in edges(net, i)] for i=1:N]
    H = [vprop(net, i)["H"] / T for i=1:N]
    
    m = zeros(N)
    m_old = zeros(N)
    TAPGraphIsing(N, adjlist, J, H, m, m_old)
end

function initrand!(g::TAPGraphIsing; μ=0, σ=1)
    for i=1:g.N
        g.m[i] = μ + σ * randn()
        g.m_old[i] = g.m[i]
    end
end

function oneTAPiter!(g::TAPGraphIsing; dump=0.1)
    @extract g: N J H m m_old
    Δ = 0.
    mnew = zeros(N)
    for i=1:N
        h = H[i]
        for (ki, j) in enumerate(g.adjlist[i])
            h += J[i][ki]*m[j] - J[i][ki]^2 * (1-m[j]^2) * m_old[i]
        end
        mnew[i] = tanh(h)
    end
    for i=1:N
        m_old[i] = m[i]
        m[i] = (1-dump) * mnew[i] + dump * m_old[i]
        Δ = max(Δ, abs(m[i] - m_old[i]))
    end
    return Δ
end

function converge!(g::TAPGraphIsing; maxiters::Int=1000, 
                                    ϵ::Float64=1e-6, 
                                    verbose::Bool=false, 
                                    dump=0)
    for it=1:maxiters
        verbose && print("it=$it ... ")
        Δ = oneTAPiter!(g; dump)
        verbose && @printf("Δ=%.2g \n", Δ)
        if Δ < ϵ
            verbose && println("Converged!")
            break
        end
    end
end

function corr_disc_nn(g::TAPGraphIsing, i::Int, j::Int)
    @extract g: N adjlist m
    ki = findfirst(adjlist[i], j)
    kj = findfirst(adjlist[j], i)
    @assert(ki > 0)
    @assert(kj > 0)

    J = g.J[i][ki]
    mi = mag(g, i)
    mj = mag(g, j)

    c = J * (1-mi^2)*(1-mj^2) / (1+ J * mi * mj) + mi*mj
    return c
end

corr_conn_nn(g::TAPGraphIsing, i::Int, j::Int) = corr_disc_nn(g, i, j) - mag(g, i)*mag(g,j)


function corr_disc_nn(g::TAPGraphIsing)
    corrs = Vector{Vector{Float64}}(g.N)
    for i=1:g.N
        corrs[i] = zeros(length(g.adjlist[i]))
    end

    for i=1:g.N
        for (ki,j) in enumerate(g.adjlist[i])
            corrs[i][ki] = corr_disc_nn(g,i,j)
        end
    end
    corrs
end

function run_tap(net::Network; T=1,  
                            maxiters=1000, 
                            ϵ=1e-6, 
                            verbose=true,
                            dump=0)
    g = TAPGraphIsing(net; T)
    initrand!(g, μ=0, σ=1)
    converge!(g; maxiters, ϵ, verbose, dump)
    return g
end
