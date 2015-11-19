type CNF
    N::Int
    M::Int
    clauses::Vector{Vector{Int}}
    allcaluses::Vector{Int}
end

function CNF(N::Int, k::Int, α::Float64; seed::Int=-1)
    if seed > 0
        srand(seed)
    end
    M = round(Int, N*α)
    allclauses = Array(Int, M * k)
    clauses = Array(Vector{Int}, M)
    for a = 1:M
        clauses[a] = pointer_to_array(pointer(allclauses, (a-1) * k + 1), k)
    end
    #clauses = Vector{Vector{Int}}()
    for a = 1:M
        while true
            c = rand(1:N, k)
            length(union(c)) != k && continue
            c = c .* rand([-1,1], k)
            #push!(clauses, c)
            copy!(clauses[a], c)
            break
        end
    end
    return CNF(N, M, clauses, allclauses)
end

function readcnf(fname::AbstractString)
    N, M, clauses = open(fname, "r") do f
        head = split(readline(f))
        N, M = parse(Int, head[3]), parse(Int, head[4])
        clauses = Vector{Vector{Int}}()
        for i = 1:M
            line = readline(f)
            c = [parse(Int64, l) for l in split(line)[1:end-1]]
            push!(clauses, c)
        end
        return N, M, clauses
    end
    return CNF(N, M, clauses)
end

function writecnf(fname::AbstractString, cnf::CNF)
    open(fname, "w") do f
        println(f, "p cnf $(cnf.N) $(cnf.M)")
        for c in cnf.clauses
            for i in c
                print(f, i, " ")
            end
            print(f, "0\n")
        end
    end
end
