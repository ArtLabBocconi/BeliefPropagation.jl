
function rrg_to_mcgraph(g::Network)
    A = [tuple(a...) for a in adjacency_list(g)]
    z = length(first(A))
    @assert all(length.(A) .== z)
    J = [([eprop(g, e)["J"] for e in edges(g, i)]...,) for i=1:nv(g)]
    @assert all(length.(J) .== z)
    
    LEV = tuple(Set(Iterators.flatten(J))...)
    ET = eltype(first(J))
    X = RRRMC.RRG.GraphRRG{ET,LEV,z}(A, J)
    X = RRRMC.AddFields.GraphAddFields(vprop(g, "H").data, X)
    return X
end


function run_monte_carlo(X::RRRMC.Interface.AbstractGraph;
                    β = 2.0,
                    sweeps = 10^5,
                    infotime = 10^2,
                    seed = 17,
                    force = false,
                    filesave = false,
                    verbose = false,
                    t_limit = 40.0,
                    alg = :rrr, # [:met, :bkl, :rrr, :wtm]
                    )

    # @assert alg ∈ [:met, :bkl, :rrr, :wtm]
    @assert alg == :rrr 
    runMC = RRRMC.rrrMC
    N = X.N

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
        Cv = Vector{BitVector}() #BitArray(N, samples)
        t0 = time()
        verbose && println(f, "#mctime acc E clocktime")
        
        function hook(mct, X, C, acc, E)
            t = time() - t0
            push!(Cv, copy(C.s))
            verbose && println(f, "$mct $acc $E $t")
            return t < t_limit
        end
        
        cleanup() = filesave && close(f)
        
        return hook, cleanup, Cv
    end

    @info "### SEED = $seed"

    iters = sweeps * N
    step = infotime * N
    samples = iters ÷ step

    hook, cleanup, Chistory = gen_hook(alg, seed)
    @time Es, Clast = runMC(X, β, iters, step=step, seed=seed, hook=hook)
    cleanup()
    filesave && save(gen_Cfname(alg, seedx, seed), Dict("Chistory"=>Chistory))

    return Es, Clast, Chistory
end