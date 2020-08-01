mutable struct Dropout
    drops::Dict{Int,Set{Pair{Int,Int}}}   # level => {k,μ} to drop
end
function Dropout()
    Dropout(Dict{Int,Set{Pair{Int,Int}}}())
end

function add_rand_drops!(d::Dropout, l::Int, N::Int, M::Int, ndrops::Int)
    !haskey(d.drops, l) && (d.drops[l] = Set{Pair{Int,Int}}())
    s = d.drops[l]
    for i=1:N
        for _=1:ndrops
            push!(s, i=>rand(1:M))
        end
    end
end
