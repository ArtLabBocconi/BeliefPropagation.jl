mutable struct TAPGraphIsing
    N::Int
    adjlist::Vector{Vector{Int}}
    J::Vector{Vector{Float64}}
    ξs::Vector{Vector{Float64}} # eventually for Neural Networks
    H::Vector{Float64}
    m::Vector{Float64}
    m_old::Vector{Float64}
end

deg(g::TAPGraphIsing, i::Integer) = length(g.adjlist[i])

function TAPGraphIsingRRG(N::Int, k::Int, seed_graph::Int = -1)
    g = random_regular_graph(N, k, seed=seed_graph)
    adjlist = g.fadjlist
    @assert(length(adjlist) == N)
    J = [Vector{Float64}() for i=1:N]

    for i=1:N
        @assert(length(adjlist[i]) == k)
        J[i] = zeros(length(adjlist[i]))
    end
    H = zeros(N)
    m = zeros(N)
    m_old = zeros(N)
    ξs = Vector{Vector{Float64}}()
    TAPGraphIsing(N, adjlist, J, ξs, H, m, m_old)
end

function initrandJ!(g::TAPGraphIsing; m::Float64=0., σ::Float64=1.)
    for i=1:g.N
        for (ki, j) in enumerate(g.adjlist[i])
            (i > j) && continue
            r = m + σ * randn()
            g.J[i][ki] = r
            kj = findfirst(g.adjlist[j], i)
            g.J[j][kj] = g.J[i][ki]
        end
    end
end


function initrandHopfieldJ!(g::TAPGraphIsing, P::Integer; β=1.)
    ξs = [rand([-1,1], g.N) for i=1:P]
    for i=1:g.N
        for (ki, j) in enumerate(g.adjlist[i])
            (i > j) && continue
            g.J[i][ki] = 0.
            for μ=1:P
                g.J[i][ki] += β*ξs[μ][i]* ξs[μ][j]
            end
            kj = findfirst(g.adjlist[j], i)
            g.J[j][kj] = g.J[i][ki]
        end
    end
    g.ξs = ξs
end


function initrandH!(g::TAPGraphIsing; m::Float64=0., σ::Float64=1.)
    for i=1:g.N
        g.H[i] = m + σ * randn()
    end
end

function initrandMess!(g::TAPGraphIsing; m::Float64=1., σ::Float64=1.)
    for i=1:g.N
        setMess!(g, i, m + σ * randn())
    end
end

function oneBPiter!(g::TAPGraphIsing, dump=0.1)
    @extract g N J H m m_old
    Δ = 0.
    mnew = zeros(N)
    # for i=randperm(N)
    for i=1:N
        h = H[i]
        for (ki, j) in enumerate(g.adjlist[i])
            h += J[i][ki]*m[j] - J[i][ki]^2 * (1-m[j]^2) * m_old[i]
            # h += J[i][ki]*m[j] - J[i][ki]^2 * (1-m[j]^2) * m[i]
        end
        # m_old[i] = m[i]
        # m[i] = tanh(h)
        mnew[i] = tanh(h)
        # Δ = max(Δ, abs(m[i] - m_old[i]))
    end
    for i=1:N
        m_old[i] = m[i]
        m[i] = (1-dump)*mnew[i] + dump* m_old[i]
        Δ = max(Δ, abs(m[i] - m_old[i]))
    end
    Δ
end

function converge!(g::TAPGraphIsing; maxiters::Int = 1000, ϵ::Float64=1e-6, verbose::Bool=false, dump=0.1)
    for it=1:maxiters
        if verbose
            write("it=$it ... ")
        end
        Δ = oneBPiter!(g, dump)
        if verbose
            @printf("Δ=%f \n", Δ)
        end
        if Δ < ϵ
            if verbose
                println("Converged!")
            end
            break
        end
    end
end

mag(g::TAPGraphIsing, i::Integer) = g.m[i]
mags(g::TAPGraphIsing) = g.m


function corr_disc_nn(g::TAPGraphIsing, i::Int, j::Int)
    @extract g N adjlist m
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


setH!(g::TAPGraphIsing, H) = g.H  .= H

function setMess!(g::TAPGraphIsing, m)
    g.m .= m
    g.m_old .= m
end

function setMess!(g::TAPGraphIsing, i::Integer, m::Float64)
    g.m[i] = m
    g.m_old[i] = m
end

function setJ!(g::TAPGraphIsing, J::Vector{Vector})
    @assert(g.N == length(J))
    for i=1:g.N
        @assert(length(g.adjlist[i]) == length(J[i]))
        for k=1:deg(g, i)
            g.J[i][k] = J[i][k]
        end
    end
end

function setJ!(g::TAPGraphIsing, i::Int, j::Int, J)
    ki = findfirst(g.adjlist[i],j)
    kj = findfirst(g.adjlist[j],i)
    @assert(ki > 0)
    @assert(kj > 0)
    g.J[i][ki] = J
    g.J[j][kj] = J
end

function getJ(g::TAPGraphIsing, i::Int, j::Int)
    ki = findfirst(g.adjlist[i],j)
    g.J[i][ki]
end

function magξ(g::TAPGraphIsing, μ::Int)
    q = 0.
    for i=1:g.N
        q += g.ξs[μ][i] * g.m[i]
    end
    q /= g.N
    return q
end

magsξ(g::TAPGraphIsing) = Float64[magξ(g, μ) for μ=1:length(g.ξs)]

function mainTAPIsing(; N::Int = 1000, k::Int = 4, β::Float64 = 1., maxiters::Int = 1000, ϵ::Float64=1e-6, dump=0.,
    P::Integer = 0)
    g = TAPGraphIsingRRG(N, k)
    if P <= 0
        initrandJ!(g, m=0., σ=β/√N)
        initrandMess!(g, m=1., σ=1.)
    else
        initrandHopfieldJ!(g, P; β=β/N)
        setMess!(g, g.ξs[1])
    end
    # initrandH!(g, m=0., σ=0.)
    converge!(g, maxiters=maxiters, ϵ=ϵ, verbose = true, dump=dump)
    if P > 0
        println(magsξ(g))
        println(filter(x->abs(x)>0.5, magsξ(g)))
    end
    return mean(mags(g).^2)
end
