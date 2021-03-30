import OnlineStats

function rrg_to_mcgraph(g::Network)
    # For some weird reason RRRMC stores 
    #couplings and fields with opposite sign.
    
    A = [tuple(a...) for a in adjacency_list(g)]
    z = length(first(A))
    @assert all(length.(A) .== z)
    J = [([-eprop(g, e)["J"] for e in edges(g, i)]...,) for i=1:nv(g)]
    @assert all(length.(J) .== z)
    H = -vprop(g, "H").data
    
    LEV = tuple(Set(Iterators.flatten(J))...)
    ET = eltype(first(J))
    
    X = RRRMC.RRG.GraphRRG{ET,LEV,z}(A, J)
    X = RRRMC.AddFields.GraphAddFields(H, X)
    return X
end


function run_monte_carlo(X::RRRMC.Interface.AbstractGraph;
                    β = 2.0,
                    sweeps = 10^7,
                    infotime = 10,
                    seed = -1,
                    force = false,
                    filesave = false,
                    verbose = false,
                    t_limit = 40.0,
                    alg = :met, # [:met, :bkl, :rrr, :wtm]
                    )

    # @assert alg ∈ [:met, :bkl, :rrr, :wtm]
    if alg == :rrr
        runMC = RRRMC.rrrMC
    elseif alg == :met
        runMC = RRRMC.standardMC
    elseif alg == :met
        @error "unsupported algorithms :$alg"
    end
    N = RRRMC.Interface.getN(X)

    dirname = "output_mc_beta$(β)_tmax$(t_limit)_tstep$(infotime)"
    gen_fname(alg, seed) = joinpath(dirname, "output_$(alg)_s$(seed).txt")
    gen_Cfname(alg, seed) = joinpath(dirname, "Cs_$(alg)_s$(seed).jld2")


    function gen_hook(alg, seed)
        if filesave
            isdir(dirname) || mkdir(dirname)
            fn = gen_fname(alg, seed)
            !force && isfile(fn) && error("file $fn exists")
            f = open(fn, "w")
        else
            f = stdout
        end
        # Cv = Vector{BitVector}() #BitArray(N, samples)
        μ = [OnlineStats.Mean() for _=1:N]
        t0 = time()
        verbose && println(f, "#mctime acc E clocktime")
        
        function hook(mct, X, C, acc, E)
            t = time() - t0
            σ = 2 .* C.s .- 1
            # @show μ
            OnlineStats.fit!.(μ, σ)
            # push!(Cv, copy(C.s))
            verbose && println(f, "$mct $acc $E $t")
            return t < t_limit
        end
        
        cleanup() = filesave && close(f)
        
        return hook, cleanup, μ  #, Cv
    end

    @info "### SEED = $seed"

    iters = sweeps * N
    step = infotime * N
    samples = iters ÷ step
    # @show iterate step samples infotime

    hook, cleanup, μ = gen_hook(alg, seed)
    @time Es, Clast = runMC(X, β, iters, step=step, seed=seed, hook=hook)
    cleanup()
    # filesave && save(gen_Cfname(alg, seedx, seed), Dict("Chistory"=>Chistory))
    σlast = 2 .* Clast.s .- 1
    μ = OnlineStats.value.(μ) 
    return Es, σlast, μ  #, Chistory
end
