
@testset "umap tests" begin
    @testset "constructor" begin
        @testset "argument validation tests" begin
            @test_throws ArgumentError fit(UMAP, [1. 1.]; k=0) # k error
            @test_throws ArgumentError fit(UMAP, [1. 1.]; maxoutdim=0, k=1) # n_comps error
            @test_throws ArgumentError fit(UMAP, [1. 1.; 1. 1.; 1. 1.]; k=1, min_dist = 0.) # min_dist error
        end
    end

    @testset "input type stability tests" begin
        m = fit(UMAP, rand(5, 100); layout=RandomLayout())
        @test eltype(m.graph) == Float32  ## input Float64 -> output Float32 (SimilaritySearch distances are Float32 and the embedding takes distances as input)
        @test size(m.graph) == (100, 100)
        @test size(m.embedding) == (2, 100)
        ## @test m.data === data
        @test fit(UMAP, rand(Float32, 5, 100); layout=RandomLayout()).graph isa AbstractMatrix{Float32}
    end

    @testset "layouts" begin
        model = fit(UMAP, rand(5, 100); layout=RandomLayout())
        model = fit(UMAP, rand(5, 100); layout=SpectralLayout())
        model = fit(UMAP, rand(5, 100); layout=PrecomputedLayout(rand(2, 100)))
        model = fit(UMAP, rand(5, 100); layout=KnnGraphLayout())
    end

    @testset "dimensions" begin
        A = fit(UMAP, rand(5, 100); maxoutdim=1, layout=RandomLayout())
        A = fit(UMAP, rand(5, 100); maxoutdim=2, layout=RandomLayout())
        A = fit(UMAP, rand(5, 100); maxoutdim=3, layout=RandomLayout())
    end

    @testset "reusing umap" begin
        A = fit(UMAP, rand(5, 100); maxoutdim=2, layout=RandomLayout())
        B = fit(A, 3)
        @test size(A.embedding) == (2, 100)
        @test size(B.embedding) == (3, 100)
    end

    @testset "fuzzy_simpl_set" begin
        # knns = rand(1:50, 5, 50)
        # dists = rand(5, 50)
        knns = Int32[2 3 2; 3 1 1]
        dists = Float32[1.5 .5 .5; 2. 1.5 2.]
        k = 3
        umap_graph = fuzzy_simplicial_set(knns, dists, 3, 1, 1f0)
        @test issymmetric(umap_graph)
        @test all(0. .<= umap_graph .<= 1.)
        @test size(umap_graph) == (3, 3)

        dists = convert.(Float32, dists)
        umap_graph = fuzzy_simplicial_set(knns, dists, 3, 1, 1.f0)
        @test issymmetric(umap_graph)
        @test eltype(umap_graph) == Float32

        umap_graph = fuzzy_simplicial_set(knns, dists, 200, 1, 1., true, false)
        @test all(0. .<= umap_graph .<= 1.)
        @test size(umap_graph) == (200, 3)
    end


    @testset "smooth_knn_dists" begin
        dists = [0., 1., 2., 3., 4., 5.]
        rho = 1
        k = 6
        local_connectivity = 1
        bandwidth = 1f0
        niter = 64
        sigma = smooth_knn_dist_opt_binsearch(dists, rho, k, bandwidth, niter)
        function psum(ds, r, s)
            D = 0.0
            for d in ds
                if d > 0  # removes self-reference
                    D += exp(-max(d - r, 0.0) / s)
                end
            end

            D
        end

        @test psum(dists, rho, sigma) - log2(k-1)*bandwidth < SMOOTH_K_TOLERANCE

        knn_dists = [0. 0. 0.;
                     1. 2. 3.;
                     2. 4. 5.;
                     3. 4. 5.;
                     4. 6. 6.;
                     5. 6. 10.]

        rhos = Float32[]
        sigmas = Float32[]
        
        for col in eachcol(knn_dists)
            ρ, σ = SimSearchManifoldLearning.smooth_knn_dists_vector(col, k, local_connectivity; niter, bandwidth)
            push!(rhos, ρ)
            push!(sigmas, σ)
        end
        
        @show rhos, sigmas
        
        @test rhos == [1., 2., 3.]
        PS = [psum(knn_dists[:, i], rhos[i], sigmas[i]) for i in 1:3]
        diffs = PS .- log2(k-1)  # k-1 => removes self reference
        @info PS diffs
        @test all(diffs .< SMOOTH_K_TOLERANCE)

        knn_dists = Float32[0. 0. 0.;
                            0. 1. 2.;
                            0. 2. 3.]

        rhos = Float32[]
        sigmas = Float32[]
        
        for col in eachcol(knn_dists)
            ρ, σ = SimSearchManifoldLearning.smooth_knn_dists_vector(col, 2, 1; niter, bandwidth)
            push!(rhos, ρ)
            push!(sigmas, σ)
        end
        @show :last => rhos, sigmas
        @test rhos == [0., 1., 2.]

        # this implementation only supports integer local_connectivity
        #rhos, sigmas = smooth_knn_dists(knn_dists, 2, 1.5)
        #@test rhos == [0., 1.5, 2.5]
    end

    @testset "compute_membership_strengths" begin
        knns = [1 2 3; 2 1 2]
        dists = Float32[0. 0. 0.; 2. 2. 3.]
        rhos = Float32[2., 1., 4.]
        sigmas = Float32[1., 1., 1.]
        true_rows = [1, 2, 2, 1, 3, 2]
        true_cols = [1, 1, 2, 2, 3, 3]
        true_vals = Float32[0., 1., 0., exp(-1f0), 0., 1.]
        rows, cols, vals = compute_membership_strengths(knns, dists, 1, true)
        @show rows, cols, vals
        @test rows == true_rows
        @test cols == true_cols  # sigmas and rhos are computed inside and the given values don't match with those
        #@test vals == true_vals
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
            res_embedding = optimize_embedding(graph, query_embedding, ref_embedding, n_epochs, initial_alpha, gamma, neg_sample_rate, a, b)
            @test res_embedding isa AbstractMatrix
            @test size(res_embedding) == size(query_embedding)
            @test isapprox(old_ref_embedding, ref_embedding, atol=1e-4)
        end
    end

    @testset "spectral_layout" begin
        @info "spectral_layout!"
        A = sprand(Float64, 1000, 1000, 0.01)
        B = dropzeros!(A + A' - A .* A')
        layout = spectral_layout(B, 5)
        @test layout isa Array{Float64, 2}
        @inferred spectral_layout(B, 5)
        layout32 = spectral_layout(convert(SparseMatrixCSC{Float32}, B), 5)
        @test layout32 isa Array{Float32, 2}
        @inferred spectral_layout(convert(SparseMatrixCSC{Float32}, B), 5)
    end

    @testset "initialize_embedding" begin
        @info "initialize_embedding"
        graph = Float32[5 0 1 1;
                        2 4 1 1;
                        3 6 8 8] ./10
        ref_embedding = Float32[1 2 0;
                                0 2 -1]
        #actual = [[9, 1], [8, 2], [3, -6], [3, -6]] ./10
        actual = Float32[9  8  3  3;
                         1  2 -6 -6] ./ 10
        embedding = initialize_embedding(graph, ref_embedding) 
        @info typeof(graph) => typeof(embedding)
        @test embedding isa AbstractMatrix{Float32}
        @test isapprox(embedding, actual, atol=1e-6)

        graph = Float32.(graph[:, [1,2]])
        graph[:, end] .= 0
        ref_embedding = Float32[1 2 0;
                                0 2 -1]
        #actual = Vector{Float16}[[9, 1], [0, 0]] ./10
        actual = [9 0;
                  1 0] ./ 10
        embedding = initialize_embedding(graph, ref_embedding)
        @test embedding isa AbstractMatrix{Float32}
        @test isapprox(embedding, actual, atol=1e-2)
    end

    @testset "umap predict" begin
        @info "===== umap predict ====="
        @testset "argument validation tests" begin
            m = fit(UMAP, rand(5, 30), maxoutdim=2, k=3, n_epochs=1)
            query = rand(5, 8)
            @test_throws ArgumentError predict(m, query; k=0) # k error
            @test_throws ArgumentError predict(m, query; k=15) # k error
            # @test_throws ArgumentError predict(model, query; k=1, min_dist = 0.) # min_dist error
            t = predict(m, query; k=3)
            @test size(t) == (2, 8)
        end

        @testset "transform test" begin
            m = fit(UMAP, rand(5, 30), k=3, n_epochs=1)
            embedding = predict(m, rand(5, 10), n_epochs=5, k=5)
            @test size(embedding) == (2, 10)
            @test typeof(embedding) == typeof(m.embedding)

            m = fit(UMAP, rand(Float32, 5, 30), k=3, n_epochs=1)
            embedding = @inferred predict(m, rand(5, 50), n_epochs=5, k=5)
            @test size(embedding) == (2, 50)
            @test typeof(embedding) == typeof(m.embedding)
        end
    end
end
