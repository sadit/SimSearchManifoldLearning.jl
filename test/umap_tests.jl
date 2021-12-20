
@testset "umap tests" begin
    @testset "constructor" begin
        @testset "argument validation tests" begin
            @test_throws ArgumentError UMAP_([1. 1.]; n_neighbors=0) # n_neighbors error
            @test_throws ArgumentError UMAP_([1. 1.], 0; n_neighbors=1) # n_comps error
            # @test_throws ArgumentError UMAP_([1. 1.], 2; n_neighbors=1) # n_comps error
            @test_throws ArgumentError UMAP_([1. 1.; 1. 1.; 1. 1.]; n_neighbors=1, min_dist = 0.) # min_dist error
        end
    end

    @testset "input type stability tests" begin
        umap_ = UMAP_(rand(5, 100); init=:random)
        @test eltype(umap_.graph) == Float32  ## input Float64 -> output Float32 (SimilaritySearch distances are Float32 and the embedding takes distances as input)
        @test size(umap_.graph) == (100, 100)
        @test size(umap_.embedding) == (2, 100)
        ## @test umap_.data === data
        @test UMAP_(rand(Float32, 5, 100); init=:random).graph isa AbstractMatrix{Float32}
    end

    @testset "reusing umap" begin
        umap1 = UMAP_(rand(5, 100), 2; init=:random)
        umap2 = UMAP_(umap1, 3; init=:random)
        @test size(umap1.embedding) == (2, 100)
        @test size(umap2.embedding) == (3, 100)
    end

    @testset "fuzzy_simpl_set" begin
        # knns = rand(1:50, 5, 50)
        # dists = rand(5, 50)
        knns = [2 3 2; 3 1 1]
        dists = [1.5 .5 .5; 2. 1.5 2.]
        k = 3
        umap_graph = fuzzy_simplicial_set(knns, dists, k, 3, 1, 1.)
        @test issymmetric(umap_graph)
        @test all(0. .<= umap_graph .<= 1.)
        @test size(umap_graph) == (3, 3)

        dists = convert.(Float32, dists)
        umap_graph = fuzzy_simplicial_set(knns, dists, k, 3, 1, 1.f0)
        @test issymmetric(umap_graph)
        @test eltype(umap_graph) == Float32

        umap_graph = fuzzy_simplicial_set(knns, dists, k, 200, 1, 1., false)
        @test all(0. .<= umap_graph .<= 1.)
        @test size(umap_graph) == (200, 3)
    end

    @testset "smooth_knn_dists" begin
        dists = [0., 1., 2., 3., 4., 5.]
        rho = 1
        k = 6
        local_connectivity = 1
        bandwidth = 1.
        niter = 64
        sigma = smooth_knn_dist(dists, rho, k, bandwidth, niter)
        psum(ds, r, s) = sum(exp.(-max.(ds .- r, 0.) ./ s))
        @test psum(dists, rho, sigma) - log2(k)*bandwidth < SMOOTH_K_TOLERANCE

        knn_dists = [0. 0. 0.;
                     1. 2. 3.;
                     2. 4. 5.;
                     3. 4. 5.;
                     4. 6. 6.;
                     5. 6. 10.]
        rhos, sigmas = smooth_knn_dists(knn_dists, k, local_connectivity)
        @test rhos == [1., 2., 3.]
        diffs = [psum(knn_dists[:,i], rhos[i], sigmas[i]) for i in 1:3] .- log2(6)
        @test all(diffs .< SMOOTH_K_TOLERANCE)

        knn_dists = [0. 0. 0.;
                     0. 1. 2.;
                     0. 2. 3.]
        rhos, sigmas = smooth_knn_dists(knn_dists, 2, 1)
        @test rhos == [0., 1., 2.]

        rhos, sigmas = smooth_knn_dists(knn_dists, 2, 1.5)
        @test rhos == [0., 1.5, 2.5]
    end
    
    @testset "compute_membership_strengths" begin
        knns = [1 2 3; 2 1 2]
        dists = [0. 0. 0.; 2. 2. 3.]
        rhos = [2., 1., 4.]
        sigmas = [1., 1., 1.]
        true_rows = [1, 2, 2, 1, 3, 2]
        true_cols = [1, 1, 2, 2, 3, 3]
        true_vals = [0., 1., 0., exp(-1.), 0., 1.]
        rows, cols, vals = compute_membership_strengths(knns, dists, rhos, sigmas)
        @test rows == true_rows
        @test cols == true_cols
        @test vals == true_vals
    end

    @testset "optimize_embedding" begin
        graph1 = sparse(Symmetric(sprand(6,6,0.4)))
        graph2 = sparse(sprand(5,3,0.4))
        graph3 = sparse(sprand(3,5,0.4))

        n_epochs = 1
        initial_alpha = 1f0
        min_dist = 1f0
        spread = 1
        gamma = 1f0
        neg_sample_rate = 5
        for graph in [graph1, graph2, graph3]
            ref_embedding = rand(2, size(graph, 1))
            old_ref_embedding = deepcopy(ref_embedding)
            query_embedding = rand(2, size(graph, 2))
            a, b = fit_ab(min_dist, spread)
            res_embedding = optimize_embedding(graph, query_embedding, ref_embedding, n_epochs, initial_alpha, gamma, neg_sample_rate, a, b, move_ref=false)
            @test res_embedding isa AbstractMatrix
            @test size(res_embedding) == size(query_embedding)
            @test isapprox(old_ref_embedding, ref_embedding, atol=1e-4)
        end
    end

    @testset "spectral_layout" begin
        A = sprand(10000, 10000, 0.001)
        B = dropzeros(A + A' - A .* A')
        layout = spectral_layout(B, 5)
        @test layout isa Array{Float64, 2}
        @inferred spectral_layout(B, 5)
        layout32 = spectral_layout(convert(SparseMatrixCSC{Float32}, B), 5)
        @test layout32 isa Array{Float32, 2}
        @inferred spectral_layout(convert(SparseMatrixCSC{Float32}, B), 5)
    end

    @testset "initialize_embedding" begin
        graph = [5 0 1 1;
                 2 4 1 1;
                 3 6 8 8] ./10
        ref_embedding = Float64[1 2 0;
                                0 2 -1]
        #actual = [[9, 1], [8, 2], [3, -6], [3, -6]] ./10
        actual = [9  8  3  3;
                  1  2 -6 -6] ./ 10
        embedding = initialize_embedding(graph, ref_embedding)
        @test embedding isa AbstractMatrix{Float64}
        @test isapprox(embedding, actual, atol=1e-8)

        graph = Float16.(graph[:, [1,2]])
        graph[:, end] .= 0
        ref_embedding = Float16[1 2 0;
                                0 2 -1]
        #actual = Vector{Float16}[[9, 1], [0, 0]] ./10
        actual = [9 0;
                  1 0] ./ 10
        embedding = initialize_embedding(graph, ref_embedding)
        @test embedding isa AbstractMatrix{Float16}
        @test isapprox(embedding, actual, atol=1e-2)
    end

    @testset "umap transform" begin
        @testset "argument validation tests" begin
            data = rand(5, 10)
            model = UMAP_(data, 2, n_neighbors=2, n_epochs=1)
            query = rand(5, 8)
            # @test_throws ArgumentError transform(model, rand(6, 8); n_neighbors=3) # query size error  => the SimilaritySearch's index sees objects as a blackbox
            @test_throws ArgumentError transform(model, query; n_neighbors=0) # n_neighbors error
            @test_throws ArgumentError transform(model, query; n_neighbors=15) # n_neighbors error
            # @test_throws ArgumentError transform(model, query; n_neighbors=1, min_dist = 0.) # min_dist error

            a, b = fit_ab(0.1, 1.0)
            model = UMAP_(model.graph, model.embedding, model.index, model.n_neighbors, model.knns, model.dists, a, b)
            t = transform(model, view(query, :, 1:4); n_neighbors=3)
            @test size(t) == (2, 4)
        end

        @testset "transform test" begin
            data = rand(5, 30)
            model = UMAP_(data, 2, n_neighbors=2, n_epochs=1)
            embedding = transform(model, rand(5, 10), n_epochs=5, n_neighbors=5)
            @test size(embedding) == (2, 10)
            @test typeof(embedding) == typeof(model.embedding)

            data = rand(Float32, 5, 30)
            model = UMAP_(data, 2, n_neighbors=2, n_epochs=1)
            embedding = @inferred transform(model, rand(5, 50), n_epochs=5, n_neighbors=5)
            @test size(embedding) == (2, 50)
            @test typeof(embedding) == typeof(model.embedding)
        end
    end
end
