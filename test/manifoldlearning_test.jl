@testset "SimilaritySearch for ManifoldLearning api" begin
    X, L = ManifoldLearning.scurve(segments=5)
    for t in [ExactEuclidean, ApproxEuclidean, ExactManhattan, ApproxChebyshev]  # some examples
        M = fit(Isomap, X, nntype=t, maxoutdim=2)
        Y = predict(M)
        n = size(X, 2)
        @test size(Y) == (2, n)
    end
end
