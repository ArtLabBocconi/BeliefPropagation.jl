type CNF
    N::Int
    M::Int
    clauses::Vector{Vector{Int}}
end

function CNF(N::Int, k::Int, α::Float64; seed::Int=-1)
    if seed > 0
        srand(seed)
    end
    M = round(Int, N*α)
    clauses = Vector{Vector{Int}}()
    for a=1:M
        while true
            c = rand(1:N, k)
            length(union(c)) != k && continue
            c = c .* rand([-1,1], k)
            push!(clauses, c)
            break
        end
    end
    return CNF(N, M, clauses)
end

function readcnf(fname::AbstractString)
    f = open(fname, "r")
    head = split(readline(f))
    N, M = parse(Int, head[3]), parse(Int, head[4])
    clauses = Vector{Vector{Int}}()
    for i=1:M
        line = readline(f)
        c = [parse(Int64, e) for e in split(line)[1:end-1]]
        push!(clauses, c)
    end
    return CNF(N, M, clauses)
end

function writecnf(fname::AbstractString, cnf::CNF)
    f = open(fname, "w")
    println(f, "p cnf $(cnf.N) $(cnf.M)")
    for c in cnf.clauses
        for i in c
            print(f, i, " ")
        end
        print(f, "0\n")
    end
    close(f)
end
