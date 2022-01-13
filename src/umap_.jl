# an implementation of Uniform Manifold Approximation and Projection
# for Dimension Reduction, L. McInnes, J. Healy, J. Melville, 2018.

struct UMAP_{GraphType<:AbstractMatrix, EmbeddingType<:AbstractMatrix, IndexType}
    graph::GraphType
    embedding::EmbeddingType
    index::IndexType
    n_neighbors::Int
    knns::Matrix{Int32}
    dists::Matrix{Float32}    
    a::Float32
    b::Float32

    function UMAP_(graph::GraphType, embedding::EmbeddingType, index::IndexType, n_neighbors, knns, dists, a::Real, b::Real) where {GraphType<:AbstractMatrix, EmbeddingType<:AbstractMatrix, IndexType}
        issymmetric(graph) || isapprox(graph, graph') || error("UMAP_ constructor expected graph to be a symmetric matrix")
        new{GraphType, EmbeddingType, IndexType}(graph, embedding, index, n_neighbors, knns, dists, a, b)
    end
end


const SMOOTH_K_TOLERANCE = 1e-5

"""
    umap(X::AbstractMatrix[, n_components=2]; <kwargs>) -> embedding

Embed the data `X` into a `n_components`-dimensional space. `n_neighbors` controls
how many neighbors to consider as locally connected.

See `UMAP_` for a description of keyword arguments.
"""
function umap(args...; kwargs...)
    # this is just a convenience function for now
    return UMAP_(args...; kwargs...).embedding
end

"""
    UMAP_(data_or_index, [, n_components=2]; <kwargs>) -> UMAP_ object

Create a model representing the embedding of data `X` into `n_components`-dimensional space. 
The returned model has the following fields:

- `graph`: the graph representing the fuzzy simplicial set of the manifold of `X`.
- `embedding`: the `n-component`-dimensional embedding of the data `X`.
- `data_or_index`: a data matrix or an index to search knns (see SimilaritySearch package).
- `knns`: a matrix of indices of `X` representing each point's nearest neighbors according to `metric`.
          `knns[j, i]` is the index of point i's jth nearest neighbor.
- `dists`: the respective distances of the above neighbors.
           `dists[j, i]` is the distance of point i's jth nearest neighbor.

# Keyword Arguments
- `n_neighbors::Integer = 15`: the number of neighbors to consider as locally connected. Larger values capture more global structure in the data, while small values capture more local structure.
- `metric::{SemiMetric, Symbol} = Euclidean()`: the metric to calculate distance in the input space. It is also possible to pass `metric = :precomputed` to treat `X` like a precomputed distance matrix.
- `n_epochs::Integer = 300`: the number of training epochs for embedding optimization
- `learning_rate::Real = 1`: the initial learning rate during optimization
- `init::Symbol = :spectral`: how to initialize the output embedding; valid options are `:spectral` and `:random`
- `min_dist::Real = 0.1`: the minimum spacing of points in the output embedding
- `spread::Real = 1`: the effective scale of embedded points. Determines how clustered embedded points are in combination with `min_dist`.
- `set_operation_ratio::Real = 1`: interpolates between fuzzy set union and fuzzy set intersection when constructing the UMAP graph (global fuzzy simplicial set). The value of this parameter should be between 1.0 and 0.0: 1.0 indicates pure fuzzy union, while 0.0 indicates pure fuzzy intersection.
- `local_connectivity::Integer = 1`: the number of nearest neighbors that should be assumed to be locally connected. The higher this value, the more connected the manifold becomes. This should not be set higher than the intrinsic dimension of the manifold.
- `repulsion_strength::Real = 1`: the weighting of negative samples during the optimization process.
- `neg_sample_rate::Integer = 5`: the number of negative samples to select for each positive sample. Higher values will increase computational cost but result in slightly more accuracy.
- `a = nothing`: this controls the embedding. By default, this is determined automatically by `min_dist` and `spread`.
- `b = nothing`: this controls the embedding. By default, this is determined automatically by `min_dist` and `spread`.
- `parallel = Threads.nthreads() > 1`: controls parallelism of the methods (all-knn searches and embedding opt.)
"""
function UMAP_(
    data_or_index,
    n_components::Integer = 2;
    n_neighbors::Integer = 15,
    n_epochs::Integer = 300,
    learning_rate::Real = 1f0,
    init::Symbol = :spectral,
    min_dist::Real = 0.1f0,
    spread::Real = 1f0,
    set_operation_ratio::Real = 1f0,
    local_connectivity::Integer = 1,
    repulsion_strength::Float32 = 1f0,
    neg_sample_rate::Integer = 5,
    a = nothing,
    b = nothing,
    parallel = Threads.nthreads() > 1
)
    min_dist = convert(Float32, min_dist)
    spread = convert(Float32, spread)
    set_operation_ratio = convert(Float32, set_operation_ratio)
    learning_rate = convert(Float32, learning_rate)
    repulsion_strength = convert(Float32, repulsion_strength)
    index = data_or_index isa AbstractMatrix ? ExhaustiveSearch(; db=data_or_index, dist=SqEuclidean()) : data_or_index
    n = length(index)
    n > n_neighbors > 0 || throw(ArgumentError("the number of examples must be greater than n_neighbors and n_neighbors must be greater than 0"))
    n_components > 1 || throw(ArgumentError("n_components must be greater than 1"))

    @show (n, n_neighbors, n_components)
    # argument checking
    n_epochs > 0 || throw(ArgumentError("n_epochs must be greater than 0"))
    learning_rate > 0 || throw(ArgumentError("learning_rate must be greater than 0"))
    min_dist > 0 || throw(ArgumentError("min_dist must be greater than 0"))
    0 ≤ set_operation_ratio ≤ 1 || throw(ArgumentError("set_operation_ratio must lie in [0, 1]"))
    local_connectivity > 0 || throw(ArgumentError("local_connectivity must be greater than 0"))
    println(stderr, "*** computing allknn graph")
    knns, dists = allknn(index, n_neighbors; parallel)
    println(stderr, "*** computing graph")
    graph = fuzzy_simplicial_set(knns, dists, n_neighbors, n, local_connectivity, set_operation_ratio)
    println(stderr, "*** computing embedding")
    embedding = initialize_embedding(graph, n_components, Val(init))
    a, b = fit_ab(min_dist, spread, a, b)
    embedding = optimize_embedding(graph, embedding, embedding, n_epochs, learning_rate, repulsion_strength, neg_sample_rate, a, b; parallel)
    # TODO: if target variable y is passed, then construct target graph
    #       in the same manner and do a fuzzy simpl set intersection

    UMAP_(graph, embedding, index, n_neighbors, knns, dists, a, b)
end

"""
    UMAP_(umap_::UMAP_, n_components)

Reuses a previously computed model with a different number of components
"""
function UMAP_(U::UMAP_, n_components::Integer;
        n_epochs=300,
        learning_rate::Real = 1f0,
        init::Symbol = :spectral,
        repulsion_strength::Float32 = 1f0,
        neg_sample_rate::Integer = 5,
        a = U.a,
        b = U.b,
        parallel = Threads.nthreads() > 1
    )
    learning_rate = convert(Float32, learning_rate)
    repulsion_strength = convert(Float32, repulsion_strength)

    graph = U.graph
    embedding = initialize_embedding(graph, n_components, Val(init))
    embedding = optimize_embedding(graph, embedding, embedding, n_epochs, learning_rate, repulsion_strength, neg_sample_rate, a, b; parallel)
    # TODO: if target variable y is passed, then construct target graph
    #       in the same manner and do a fuzzy simpl set intersection

    UMAP_(graph, embedding, U.index, U.n_neighbors, U.knns, U.dists, a, b)
end

"""
    transform(model::UMAP_, Q::AbstractMatrix; <kwargs>) -> embedding

Use the given model to embed new points into an existing embedding. `Q` is a matrix of some number of points (columns)
in the same space as `model.data`. The returned embedding is the embedding of these points in n-dimensional space, where
n is the dimensionality of `model.embedding`. This embedding is created by finding neighbors of `Q` in `model.embedding`
and optimizing cross entropy according to membership strengths according to these neighbors.

# Keyword Arguments
- `n_neighbors::Integer = 15`: the number of neighbors to consider as locally connected. Larger values capture more global structure in the data, while small values capture more local structure.
- `metric::{SemiMetric, Symbol} = Euclidean()`: the metric to calculate distance in the input space. It is also possible to pass `metric = :precomputed` to treat `X` like a precomputed distance matrix.
- `n_epochs::Integer = 300`: the number of training epochs for embedding optimization
- `learning_rate::Real = 1`: the initial learning rate during optimization
- `init::Symbol = :spectral`: how to initialize the output embedding; valid options are `:spectral` and `:random`
- `min_dist::Real = 0.1`: the minimum spacing of points in the output embedding
- `spread::Real = 1`: the effective scale of embedded points. Determines how clustered embedded points are in combination with `min_dist`.
- `set_operation_ratio::Real = 1`: interpolates between fuzzy set union and fuzzy set intersection when constructing the UMAP graph (global fuzzy simplicial set). The value of this parameter should be between 1.0 and 0.0: 1.0 indicates pure fuzzy union, while 0.0 indicates pure fuzzy intersection.
- `local_connectivity::Integer = 1`: the number of nearest neighbors that should be assumed to be locally connected. The higher this value, the more connected the manifold becomes. This should not be set higher than the intrinsic dimension of the manifold.
- `repulsion_strength::Real = 1`: the weighting of negative samples during the optimization process.
- `neg_sample_rate::Integer = 5`: the number of negative samples to select for each positive sample. Higher values will increase computational cost but result in slightly more accuracy.
- `a = nothing`: this controls the embedding. By default, this is determined automatically by `min_dist` and `spread`.
- `b = nothing`: this controls the embedding. By default, this is determined automatically by `min_dist` and `spread`.
"""
function transform(model::UMAP_, Q;
                   n_neighbors = model.n_neighbors,
                   n_epochs::Integer = 100,
                   learning_rate::Real = 1.0,
                   set_operation_ratio::Real = 1.0,
                   local_connectivity::Integer = 1,
                   repulsion_strength::Real = 1.0,
                   neg_sample_rate::Integer = 5,
                   a = model.a,
                   b = model.b,
                   parallel = Threads.nthreads() > 1
                   ) where {S<:Real}
    
    set_operation_ratio = convert(Float32, set_operation_ratio)
    learning_rate = convert(Float32, learning_rate)
    repulsion_strength = convert(Float32, repulsion_strength)
    Q = convert(AbstractDatabase, Q)

    # argument checking
    length(model.index) > n_neighbors > 0            || throw(ArgumentError("n_neighbors must be greater than 0"))
    learning_rate > 0                                || throw(ArgumentError("learning_rate must be greater than 0"))
    0 ≤ set_operation_ratio ≤ 1                      || throw(ArgumentError("set_operation_ratio must lie in [0, 1]"))
    local_connectivity > 0                           || throw(ArgumentError("local_connectivity must be greater than 0"))
    length(model.index) == size(model.embedding, 2)  || throw(ArgumentError("model.index must have same number of columns as model.embedding"))
    #size(model.data, 1) == size(Q, 1)                || throw(ArgumentError("size(model.data, 1) must equal size(Q, 1)"))
    
    n_epochs = max(0, n_epochs)
    # main algorithm
    n = length(model.index)
    knns, dists = searchbatch(model.index, Q, n_neighbors; parallel)
    graph = fuzzy_simplicial_set(knns, dists, n_neighbors, n, local_connectivity, set_operation_ratio, false)

    E = initialize_embedding(graph, model.embedding)
    optimize_embedding(graph, E, model.embedding, n_epochs, learning_rate, repulsion_strength, neg_sample_rate, a, b; parallel)
end

"""
    fuzzy_simplicial_set(knns, dists, n_neighbors, n_points, local_connectivity, set_op_ratio, apply_fuzzy_combine=true) -> membership_graph::SparseMatrixCSC, 

Construct the local fuzzy simplicial sets of each point represented by its distances
to its `n_neighbors` nearest neighbors, stored in `knns` and `dists`, normalizing the distances
on the manifolds, and converting the metric space to a simplicial set.
`n_points` indicates the total number of points of the original data, while `knns` contains
indices of some subset of those points (ie some subset of 1:`n_points`). If `knns` represents
neighbors of the elements of some set with itself, then `knns` should have `n_points` number of
columns. Otherwise, these two values may be inequivalent.
If `apply_fuzzy_combine` is true, use intersections and unions to combine
fuzzy sets of neighbors (default true).

The returned graph will have size (`n_points`, size(knns, 2)).
"""
function fuzzy_simplicial_set(knns,
                              dists,
                              n_neighbors,
                              n_points,
                              local_connectivity,
                              set_operation_ratio,
                              apply_fuzzy_combine=true)

    σs, ρs = smooth_knn_dists(dists, n_neighbors, local_connectivity)

    rows, cols, vals = compute_membership_strengths(knns, dists, σs, ρs)
    fs_set = sparse(rows, cols, vals, n_points, size(knns, 2))

    if apply_fuzzy_combine
        dropzeros(combine_fuzzy_sets(fs_set, convert(eltype(fs_set), set_operation_ratio)))
    else
        dropzeros(fs_set)
    end
end

"""
    smooth_knn_dists(dists, k, local_connectivity; <kwargs>) -> knn_dists, nn_dists

Compute the distances to the nearest neighbors for a continuous value `k`. Returns
the approximated distances to the kth nearest neighbor (`knn_dists`)
and the nearest neighbor (nn_dists) from each point.
"""
function smooth_knn_dists(knn_dists::AbstractMatrix{S},
                          k::Real,
                          local_connectivity::Real;
                          niter::Integer=64,
                          bandwidth::Real=1) where {S <: Real}

    nonzero_dists(dists) = @view dists[dists .> 0.]
    ρs = zeros(S, size(knn_dists, 2))
    σs = Array{S}(undef, size(knn_dists, 2))
    for i in 1:size(knn_dists, 2)
        nz_dists = nonzero_dists(knn_dists[:, i])
        if length(nz_dists) >= local_connectivity
            index = floor(Int, local_connectivity)
            interpolation = local_connectivity - index
            if index > 0
                ρs[i] = nz_dists[index]
                if interpolation > SMOOTH_K_TOLERANCE
                    ρs[i] += interpolation * (nz_dists[index+1] - nz_dists[index])
                end
            else
                ρs[i] = interpolation * nz_dists[1]
            end
        elseif length(nz_dists) > 0
            ρs[i] = maximum(nz_dists)
        end
        @inbounds σs[i] = smooth_knn_dist(view(knn_dists, :, i), ρs[i], k, bandwidth, niter)
    end

    return ρs, σs
end

# calculate sigma for an individual point
@fastmath function smooth_knn_dist(dists::AbstractVector, ρ, k, bandwidth, niter)
    target = log2(k)*bandwidth
    lo, mid, hi = 0., 1., Inf
    for n in 1:niter
        psum = sum(exp.(-max.(dists .- ρ, 0.)./mid))
        if abs(psum - target) < SMOOTH_K_TOLERANCE
            break
        end
        if psum > target
            hi = mid
            mid = (lo + hi)/2.
        else
            lo = mid
            if hi == Inf
                mid *= 2.
            else
                mid = (lo + hi) / 2.
            end
        end
    end
    # TODO: set according to min k dist scale
    return mid
end

"""
    compute_membership_strengths(knns, dists, σs, ρs) -> rows, cols, vals

Compute the membership strengths for the 1-skeleton of each fuzzy simplicial set.
"""
function compute_membership_strengths(knns::AbstractMatrix{S},
                                      dists::AbstractMatrix{T},
                                      ρs::Vector{T},
                                      σs::Vector{T}) where {S <: Integer, T}
    # set dists[i, j]
    rows = sizehint!(S[], length(knns))
    cols = sizehint!(S[], length(knns))
    vals = sizehint!(T[], length(knns))
    for i in 1:size(knns, 2), j in 1:size(knns, 1)
        @inbounds if i == knns[j, i] # dist to self
            d = 0.
        else
            @inbounds d = exp(-max(dists[j, i] - ρs[i], 0.)/σs[i])
        end
        append!(cols, i)
        append!(rows, knns[j, i])
        append!(vals, d)
    end
    return rows, cols, vals
end
