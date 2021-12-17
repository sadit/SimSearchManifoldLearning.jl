@testset "utils tests" begin
    @testset "combine_fuzzy_sets tests" begin
        A = [1.0 0.1; 0.4 1.0]
        union_res = [1.0 0.46; 0.46 1.0]
        res = combine_fuzzy_sets(A, 1.0)
        @test isapprox(res, union_res)
        inter_res = [1.0 0.04; 0.04 1.0]
        res = combine_fuzzy_sets(A, 0.0)
        @test isapprox(res, inter_res)

    end

    @testset "fit_ab tests" begin
        @test all((1, 2) .== fit_ab(-1, 0, 1, 2))
    end
end
