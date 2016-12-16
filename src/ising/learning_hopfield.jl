function make_new_learning_weights!(g::FactorGraphIsing, H::Vector, maxiters_learning, η, λ)
    for it=1:maxiters_learning
        setH!(g, λ*H)
        converge!(g)
        cdiscλ = corr_disc_nn(g)

        setH!(g, 0.)
        converge!(g)
        cdisc = corr_disc_nn(g)

        for i=1:g.N
            for (ki,j) in enumerate(g.adjlist[i])
                i > j && continue
                ΔJ = η*(cdiscλ[i][ki] - cdisc[i][ki])
                J = getJ(g, i, j)
                setJ!(g, i, j, J + ΔJ)
            end
        end
    end
end
function q_retrieval(g, ξ, βmc, niters_mc, pflip)
    σ = deepcopy(ξ)
    N = g.N
    for i=1:N
        if rand() < pflip
            σ[i] *= -1
        end
    end

    ising_mc!(σ, g.J, g.adjlist, niters=niters_mc, β=βmc)
    q = 0.
    it = 0
    while it < div(niters_mc, 10)
        ising_mc!(σ, g.J, g.adjlist, niters=1, β=βmc)
        q += dot(ξ, σ) / N
        it +=1
    end
    return q / it
end

function main_learning(;N::Int=1000, k::Int=16
                , P::Int=16, seed_ξ::Int = -1
                , maxiters_learning::Int=3, η::Float64=0.02
                , λs::Vector{Float64} = [0.2], TL::Int=10
                , βmc::Float64=2., niters_mc::Int=100, pflip::Float64=0.1)

        ξs = [rand([-1,1], N) for i=1:P]
        g = FactorGraphIsingRRG(N, k)
        num_retrieved = Vector{Vector{Float64}}()
        for (iλ,λ) in enumerate(λs)
            initrandJ!(g, m=0., σ=0.)
            initrandH!(g, m=0., σ=0.)
            initrandMess!(g, m=0., σ=0.)
            for t=1:TL
                for μ in 1:P
                    make_new_learning_weights!(g, ξs[μ], maxiters_learning, η, λ)
                end
            end

            nretr = 0
            for μ=1:P
                q = q_retrieval(g, ξs[μ], βmc, niters_mc, pflip)
                println("q$μ=$q")
                if q >= 0.6
                    nretr += 1
                end
            end
            println("λ=$λ nretr=$nretr")
            push!(num_retrieved, [λ, nretr])
        end
        return g, num_retrieved
end
