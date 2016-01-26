
function mc_ising!(σ::Vector{Int}, J, adj::Vector{Vector{Int}}
            ; β::Float64=1., niters::Int=1000)

    N = length(J)
    for it=1:niters
        for i=randperm(N)
            heff = 0.
            for (k,j) in enumerate(adj[i])
                heff += J[i][k] * σ[j]
            end
            dE = 2*σ[i]*heff
            if rand() < exp(-β*dE)
                σ[i] *= -1
            end
        end
    end
end
