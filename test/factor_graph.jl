g = FactorGraph(3, 4)
@test nv(g) == 7
@test ne(g) == 0
@test_throws ArgumentError add_edge!(g, 1, 2)
@test_throws ArgumentError add_edge!(g, 6, 7)
add_edge!(g, 1, 4)
@test ne(g) == 1

g = FactorGraph(3, 4, 8)
@test g.nvars == 3
@test g.nfacts == 4
@test nv(g) == 7
@test ne(g) == 8

g = BeliefPropagation.random_bipartite_regular_graph(5, 6, 2)
degs = degree(g)
@test all(degs[1:5] .== 2)

g = BeliefPropagation.random_bipartite_regular_graph(5, 6, 2, first_regular=false)
degs = degree(g)
@test all(degs[6:11] .== 2)
