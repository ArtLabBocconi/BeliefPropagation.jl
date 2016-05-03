using BeliefPropagation.DeepBinary: GH, G, H, GHnaive
for x=-30:1.:100
    @test_approx_eq_eps(log(GH(x)), log(G(big(x))/H(big(x))), 1e-8)
end

for x=-34:1.:34
    # println("x=$x")
    # @test_approx_eq_eps(log(GH(1,x)), log(G(big(x))/H(big(x))), 1e-8)
    @test_approx_eq_eps(log(GH(1,x)), log(G(big(x))/H(big(x))), 1e-8)
    @test_approx_eq_eps(log(-GH(0.,x)), log(G(big(-x))/H(big(-x))), 1e-8)
end


# for p in union(-1:0.001:-0.99,-0.99:0.01:0.99,0.99:0.001:1)
for p in union(0.5:0.01:0.99,0.99:0.001:0.999)
    for x=-34:1.:34
        # println("p=$p x=$x")
        # @test_approx_eq_eps(log(GH(p, x)), log(GH(big(p), big(x))), 1e0)
        @test_approx_eq_eps(log(GH(p, x)), log(GHnaive(big(p), big(x))), 1e0)
    end
end

for p in union(0.5:-0.01:0.01, 0.01:-0.001:0.001)
    for x=-34:1.:34
        # println("p=$p x=$x")
        # @test_approx_eq_eps(log(GH(p, x)), log(GH(big(p), big(x))), 1e0)
        @test_approx_eq_eps(log(-GH(p, x)), log(-GHnaive(big(p), big(x))), 1e0)
    end
end
