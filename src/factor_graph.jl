mutable struct FactorGraph <: ANetwork
    nvars::Int                        
    nfacts::Int                          
    ne::Int
    edge_index_range::Int
    out_edges::Vector{Vector{Pair{Int,Int}}}  # Unordered adjlist
    epos::Vector{Pair{Int,Int}}      # Position of the edge in out_edges
    free_indexes::Vector{Int}       # Indexes of deleted edges
    props::PropertyStore
end

function FactorGraph(nvars::Int=0, nfacts::Int=0)
    n = nvars + nfacts
    out_edges = [Vector{Pair{Int,Int}}() for _=1:n]
    epos = Vector{Pair{Int,Int}}()
    free_indexes = Vector{Int}()
    return FactorGraph(nvars, nfacts, 0, 0, out_edges, epos, free_indexes, PropertyStore())
end

FactorGraph(nvars::Int, nfacts::Int, nedges::Int) = bipartite_erdos_renyi(nvars, nfacts, nedges)

edgetype(::Type{G}) where {G<:FactorGraph} = IndexedEdge
vertextype(::Type{G}) where {G<:FactorGraph} = Int

nv(g::FactorGraph) = length(g.out_edges)
ne(g::FactorGraph) = g.ne

variables(g::FactorGraph) = 1:g.nvars
factors(g::FactorGraph) = g.nvars+1:nv(g)

# TODO can be improved (see graph/digraph)
function edge(g::FactorGraph, i::Integer, j::Integer)
    (i > nv(g) || j > nv(g)) && return IndexedEdge(i, j, -1)
    oes = g.out_edges[i]
    pos = findfirst(e->e.first==j, oes)
    if pos !== nothing
        return IndexedEdge(i, j, oes[pos].second)
    else
        return IndexedEdge(i, j, -1)
    end
end

function out_edges(g::FactorGraph, i::Integer)
    oes = g.out_edges[i]
    return (IndexedEdge(i, j, idx) for (j, idx) in oes)
end

function out_neighbors(g::FactorGraph, i::Integer)
    oes = g.out_edges[i]
    return (j for (j, idx) in oes)
end

pop_vertex!(g::FactorGraph) = (clean_vertex!(g, nv(g)); pop!(g.out_edges); nv(g)+1)

# function add_vertex!(g::FactorGraph)
#     push!(g.out_edges, Vector{Pair{Int,Int}}())
#     return nv(g)
# end

function add_factor!(g::FactorGraph)
    push!(g.out_edges, Vector{Pair{Int,Int}}())
    g.nfacts += 1
    return nv(g)
end

function add_variable!(g::FactorGraph)
    push!(g.out_edges, Vector{Pair{Int,Int}}())
    g.nvars += 1
    swap_vertices!(g, g.nvars, nv(g))
    return g.nvars
end

function add_edge!(g::FactorGraph, u::Integer, v::Integer)
    (u ∈ variables(g) && v ∈ factors(g)) ||
    (v ∈ variables(g) && u ∈ factors(g)) ||
        throw(ArgumentError("Can add edge only between variable and factor."))  
    u, v = u <= v ? (u, v) : (v, u)
    (u in vertices(g) && v in vertices(g)) || return (false, IndexedEdge(u,v,-1))
    has_edge(g, u, v) && return (false, IndexedEdge(u,v,-1)) # could be removed for multigraphs

    if isempty(g.free_indexes)
        g.edge_index_range += 1
        idx = g.edge_index_range
    else
        idx = pop!(g.free_indexes)
    end
    oes = g.out_edges[u]
    ies = g.out_edges[v]
    push!(oes, Pair(v, idx))
    if u != v
        push!(ies, Pair(u, idx))
    end
    g.ne += 1

    length(g.epos) < idx && resize!(g.epos, idx)
    g.epos[idx] = Pair(length(oes), length(ies))

    return (true, IndexedEdge(u,v,idx))
end

rem_edge!(g::FactorGraph, s::Integer, t::Integer) = rem_edge!(g, edge(g, s, t))

function rem_edge!(g::FactorGraph, e::IndexedEdge)
    s = e.src
    t = e.dst
    if s > t
        s,t = t,s
    end
    idx = e.idx
    idx <= 0 && return false
    oes = g.out_edges[s]
    ies = g.out_edges[t]
    idx > length(g.epos) && return false
    length(oes) == 0 && return false
    p1 = g.epos[idx].first
    p1 < 0 && return false

    back = last(oes)
    if back.first > s
        p2 = g.epos[back.second].second
        g.epos[back.second] = Pair(p1 , p2)
    elseif back.first == s #fix self-edges
        g.epos[back.second] = Pair(p1, p1)
    else
        p2 = g.epos[back.second].first
        g.epos[back.second] = Pair(p2 , p1)
    end
    oes[p1] = back
    pop!(oes)

    if s != t
        back = last(ies)
        p1 = g.epos[idx].second
        if back.first > t
            p2 = g.epos[back.second].second
            g.epos[back.second] = Pair(p1 , p2)
        elseif back.first == t #fix self-edges
            g.epos[back.second] = Pair(p1 , p1)
        else
            p2 = g.epos[back.second].first
            g.epos[back.second] = Pair(p2 , p1)
        end
        ies[p1] = back
        pop!(ies)
    end

    g.epos[idx] = Pair(-1,-1)

    g.ne -= 1
    push!(g.free_indexes, idx)
    return true
end

function in_edges(g::FactorGraph, i::Integer)
    ies = g.out_edges[i]
    return (IndexedEdge(j, i, idx) for (j, idx) in ies)
end

function swap_vertices!(g::FactorGraph, u::Integer, v::Integer)
    if u != v
        #TODO copying to avoid problems with self edges
        # maybe can copy only one of the two
        neigu = deepcopy(g.out_edges[u])
        neigv = deepcopy(g.out_edges[v])

        for (k,p) in enumerate(neigu)
            j, idx = p
            kj = j <= u ? g.epos[idx].first : g.epos[idx].second
            g.out_edges[j][kj] = Pair(v, idx)
            g.epos[idx] = j <= v ? Pair(kj, k) : Pair(k, kj)
        end

        for (k,p) in enumerate(neigv)
            j, idx = p
            kj = j <= v ? g.epos[idx].first : g.epos[idx].second
            g.out_edges[j][kj] = Pair(u, idx)
            g.epos[idx] = j <= u ? Pair(kj, k) : Pair(k, kj)
        end

        g.out_edges[u], g.out_edges[v] = g.out_edges[v], g.out_edges[u]

        swap_vertices!(g.props, u, v)
    end
end

function bipartite_erdos_renyi(n1::Int, n2::Int, p::Real; seed::Int=-1)
    m = n1 * n2
    rng = Erdos.getRNG(seed)
    nedg = rand(rng, Binomial(m, p))
    seed = seed >= 0 ? seed + 1 : seed
    return erdos_renyi(n1, n2, nedg; seed)
end

function bipartite_erdos_renyi(n1::Int, n2::Int, m::Int; seed::Int = -1)
    m > n1 * n2 && throw(ArgumentError("Cannot insert more than n1*n2 edges."))
    rng = Erdos.getRNG(seed)
    g = FactorGraph(n1, n2)
    while ne(g) < m
        source = rand(rng, 1:n1)
        dest = rand(rng, n1+1:n1+n2)
        res = add_edge!(g, source, dest)
    end
    return g
end

function random_bipartite_regular_graph(n1::Int, n2::Int, k::Int; 
                                    seed::Int = -1, first_regular=true)
    
    if (first_regular && k > n2) || (!first_regular && k > n1)
        throw(ArgumentError("Not enough vertices for the required degree."))
    end

    rng = Erdos.getRNG(seed)
    g = FactorGraph(n1, n2)
    if first_regular
        for i in 1:n1
            while degree(g, i) < k
                j = rand(rng, n1+1:n1+n2)
                add_edge!(g, i, j)
            end
        end
    else
        for j in n1+1:n1+n2
            while degree(g, j) < k
                i = rand(rng, 1:n1)
                add_edge!(g, i, j)
            end
        end
    end
    return g
end

