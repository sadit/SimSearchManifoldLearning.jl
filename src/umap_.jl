# an implementation of Uniform Manifold Approximation and Projection
# for Dimension Reduction, L. McInnes, J. Healy, J. Melville, 2018.

export UMAP, optimize_embedding!

"""
    struct UMAP

The UMAP model struct

# Properties

- `graph`: The fuzzy simplicial set that represents the all knn graph
- `embedding`: The embedding projection 
- `k` the number of neighbors used to create the model
- `a` and `b`: parameters to ensure an well distributed and smooth projection (from `min_dist` and `spread` arguments in `fit`)
- `index`: the search index, it can be nothing if the model is handled directly with precomputed `knns` and `dists` matrices
"""
struct UMAP
    graph::SparseMatrixCSC{Float32,Int32}
    embedding::Matrix{Float32}
    k::Int
    a::Float32
    b::Float32
    index::Union{AbstractSearchIndex,Nothing}   # this object is used with function barriers
end

const SMOOTH_K_TOLERANCE = 1e-5
const ZERO_DISTANCE_THRESHOLD = 1e-6

"""
    fit(Type{UMAP}, knns, dists; <kwargs>) -> UMAP object

Create a model representing the embedding of data `(X, dist)` into `maxoutdim`-dimensional space.
Note that `knns` and `dists` jointly specify the all `k` nearest neighbors of ``(X, dist)``,
these results must not include self-references. See the `allknn` method in `SimilaritySearch`.

# Arguments

- `knns`: A ``(k, n)`` matrix of integers (identifiers).
- `dists`: A ``(k, n)`` matrix of floating points (distances).

It uses all available threads for the projection.

# Keyword Arguments
- `maxoutdim::Integer=2`: The number of components in the embedding
- `n_epochs::Integer = 300`: the number of training epochs for embedding optimization
- `learning_rate::Real = 1`: the initial learning rate during optimization
- `learning_rate_decay::Real = 0.9`: how much `learning_rate` is updated on each epoch `(learning_rate *= learning_rate_decay)` (a minimum value is also considered as 1e-6)
- `layout::AbstractLayout = SpectralLayout()`: how to initialize the output embedding
- `min_dist::Real = 0.1`: the minimum spacing of points in the output embedding
- `spread::Real = 1`: the effective scale of embedded points. Determines how clustered embedded points are in combination with `min_dist`.
- `set_operation_ratio::Real = 1`: interpolates between fuzzy set union and fuzzy set intersection when constructing the UMAP graph (global fuzzy simplicial set). The value of this parameter should be between 1.0 and 0.0: 1.0 indicates pure fuzzy union, while 0.0 indicates pure fuzzy intersection.
- `local_connectivity::Integer = 1`: the number of nearest neighbors that should be assumed to be locally connected. The higher this value, the more connected the manifold becomes. This should not be set higher than the intrinsic dimension of the manifold.
- `repulsion_strength::Real = 1`: the weighting of negative samples during the optimization process.
- `neg_sample_rate::Integer = 5`: the number of negative samples to select for each positive sample. Higher values will increase computational cost but result in slightly more accuracy.
- `tol::Real = 1e-4`: tolerance to early stopping while optimizing embeddings.
- `minbatch=0`: controls how parallel computation is made, zero to use `SimilaritySearch` defaults and -1 to avoid parallel computation; passed to `@batch` macro of `Polyester` package.
- `a = nothing`: this controls the embedding. By default, this is determined automatically by `min_dist` and `spread`.
- `b = nothing`: this controls the embedding. By default, this is determined automatically by `min_dist` and `spread`.

"""
function fit(::Type{UMAP},
    knns::Matrix{<:Integer},
    dists::Matrix{<:AbstractFloat};
    maxoutdim::Integer = 2,
    n_epochs::Integer = 50,
    learning_rate::Real = 1f0,
    learning_rate_decay::Real = 0.9f0,
    layout::AbstractLayout = SpectralLayout(),
    min_dist::Real = 0.1f0,
    spread::Real = 1f0,
    set_operation_ratio::Real = 1f0,
    local_connectivity::Integer = 1,
    repulsion_strength::Float32 = 1f0,
    neg_sample_rate::Integer = 5,
    tol=1e-4,
    minbatch::Integer = 0
)
    min_dist = convert(Float32, min_dist)
    spread = convert(Float32, spread)
    set_operation_ratio = convert(Float32, set_operation_ratio)
    learning_rate = convert(Float32, learning_rate)
    learning_rate_decay = convert(Float32, learning_rate_decay)
    repulsion_strength = convert(Float32, repulsion_strength)
    size(knns) == size(dists) || throw(ArgumentError("knns and dists must have the same size"))
    maxoutdim > 0 || throw(ArgumentError("maxoutdim must be greater than 0"))

    # argument checking
    n_epochs > 0 || throw(ArgumentError("n_epochs must be greater than 0"))
    learning_rate > 0 || throw(ArgumentError("learning_rate must be greater than 0"))
    min_dist > 0 || throw(ArgumentError("min_dist must be greater than 0"))
    0 ≤ set_operation_ratio ≤ 1 || throw(ArgumentError("set_operation_ratio must lie in [0, 1]"))
    local_connectivity > 0 || throw(ArgumentError("local_connectivity must be greater than 0"))

    n_neighbors, n = size(knns)
    println(stderr, "*** computing graph")
    timegraph = @elapsed graph = fuzzy_simplicial_set(knns, dists, n, local_connectivity, set_operation_ratio, true; minbatch)
    println(stderr, "*** layout embedding $(typeof(layout))")
    timeinit = @elapsed embedding = initialize_embedding(layout, graph, knns, dists, maxoutdim)
    println(stderr, "*** fit ab / embedding")
    a, b = fit_ab(min_dist, spread, nothing, nothing)
    println(stderr, "*** opt embedding")
    timeopt = @elapsed embedding = optimize_embedding(graph, embedding, embedding, n_epochs, learning_rate, repulsion_strength, neg_sample_rate, a, b; minbatch, tol, learning_rate_decay)
    # TODO: if target variable y is passed, then construct target graph
    #       in the same manner and do a fuzzy simpl set intersection
    println(stderr,
    """
UMAP construction time cost report:
- fuzzy graph: $timegraph
- embedding init: $timeinit
- embedding opt: $timeopt
""")
    UMAP(graph, embedding, n_neighbors, a, b, nothing)
end

"""
    fit(::Type{<:UMAP}, index_or_data;
        k=15,
        dist::SemiMetric=L2Distance,
        minbatch=0,
        kwargs...)

Wrapper for `fit` that computes `n_nearests` nearest neighbors on `index_or_data` and passes these and `kwargs` to regular `fit`.

# Arguments

- `index_or_data`: an already constructed index (see `SimilaritySearch`), a matrix, or an abstact database (`SimilaritySearch`)

# Keyword arguments
- `k=15`: number of neighbors to compute
- `dist=L2Distance()`: A distance function (see `Distances.jl`)
- `searchctx`: search context (hyperparameters, caches, etc)
"""
function fit(
        t::Type{<:UMAP},
        index_or_data::Union{<:AbstractSearchIndex,<:AbstractDatabase,<:AbstractMatrix};
        k::Integer=15,
        dist::SemiMetric=L2Distance(),
        searchctx=nothing,
        minbatch = 0,
        kwargs...
        )
    index = if index_or_data isa AbstractMatrix
        db = convert(AbstractDatabase, index_or_data)
        ExhaustiveSearch(; dist, db)
    elseif index_or_data isa AbstractDatabase
        ExhaustiveSearch(; dist, db=index_or_data)
    else
        index_or_data
    end

    0 < k < length(index) || throw(ArgumentError("number of neighbors must be in 0 < k < number of points"))

    searchctx = searchctx === nothing ? getcontext(index) : searchctx
    @time knns = allknn(index, searchctx, k)
	knns_ = Matrix{UInt32}(undef, size(knns)...)
	dists_ = Matrix{Float32}(undef, size(knns)...)
	for i in CartesianIndices(knns)
		p = knns[i]
		knns_[i] = p.id
		dists_[i] = p.weight
	end
    m = fit(t, knns_, dists_; minbatch, kwargs...)
    UMAP(m.graph, m.embedding, m.k, m.a, m.b, index)
end

"""
    optimize_embedding!(model::UMAP; <kwargs>)

Improves the internal embedding of the model refining with more epochs

# Keyword arguments
- `n_epochs=50`.
- `learning_rate::Real = 0.1f0`.
- `learning_rate_decay::Real = 0.9f0`.
- `repulsion_strength::Float32 = 1f0`.
- `neg_sample_rate::Integer = 5`.
- `tol::Real = 1e-4`: tolerance to early stopping while optimizing embeddings.
- `minbatch=0`: controls how parallel computation is made. See [`SimilaritySearch.getminbatch`](@ref) and `@batch` (`Polyester` package).
"""
function optimize_embedding!(U::UMAP;
        n_epochs=50,
        learning_rate::Real = 0.1f0,
        learning_rate_decay::Real = 0.9f0,
        repulsion_strength::Float32 = 1f0,
        neg_sample_rate::Integer = 5,
        tol=1e-4,
        minbatch::Integer = 0
    )
    optimize_embedding(U.graph, U.embedding, U.embedding, n_epochs, learning_rate, repulsion_strength, neg_sample_rate, U.a, U.b; tol, learning_rate_decay, minbatch)
    U
end

"""
    fit(UMAP::UMAP, maxoutdim; <kwargs>)

Reuses a previously computed model with a different number of components

# Keyword arguments

- `n_epochs=50`: number of epochs to run
- `learning_rate::Real = 1f0`: initial learning rate
- `learning_rate_decay::Real = 0.9f0`: how learning rate is adjusted per epoch `learning_rate *= learning_rate_decay`
- `repulsion_strength::Float32 = 1f0`: repulsion force (for negative sampling)
- `neg_sample_rate::Integer = 5`: how many negative examples per object are used.
- `tol::Real = 1e-4`: tolerance to early stopping while optimizing embeddings.
- `minbatch=0`: controls how parallel computation is made. See [`SimilaritySearch.getminbatch`](@ref) and `@batch` (`Polyester` package).
"""
function fit(
        U::UMAP, maxoutdim::Integer;
        n_epochs=50,
        learning_rate::Real = 1f0,
        learning_rate_decay::Real = 0.9f0,
        repulsion_strength::Float32 = 1f0,
        neg_sample_rate::Integer = 5,
        graph = U.graph,
        minbatch=0,
        tol=1e-4,
        a = U.a,
        b = U.b
    )

    k, n = size(U.embedding)
    embedding = if k >= maxoutdim
        U.embedding[1:maxoutdim, :]
    else
        vcat(U.embedding, rand(-10f0:eps(Float32):10f0, maxoutdim-k, n))
    end

    learning_rate = convert(Float32, learning_rate)
    learning_rate_decay = convert(Float32, learning_rate_decay)
    repulsion_strength = convert(Float32, repulsion_strength)
    embedding = optimize_embedding(graph, embedding, embedding, n_epochs, learning_rate, repulsion_strength, neg_sample_rate, a, b; tol, learning_rate_decay, minbatch)
    UMAP(graph, embedding, U.k, a, b, U.index)
end

"""
    predict(model::UMAP)

Returns the internal embedding (the entire dataset projection)
"""
predict(model::UMAP) = model.embedding

"""
    predict(model::UMAP, Q::AbstractDatabase; k::Integer=15, kwargs...)
    predict(model::UMAP, knns, dists; <kwargs>) -> embedding

Use the given model to embed new points ``Q`` into an existing embedding produced by ``(X, dist)``.
The second function represent `Q` using its `k` nearest neighbors in `X` under some distance function (`knns` and `dists`)
See `searchbatch` in `SimilaritySearch` to compute both (also for `AbstractDatabase` objects).

# Arguments
- `model`: The fitted model
- `knns`: matrix of identifiers (integers) of size ``(k, |Q|)``
- `dists`: matrix of distances (floating point values) of size ``(k, |Q|)``

Note: the number of neighbors `k` (embedded into knn matrices) control the embedding. Larger values capture more global structure in the data, while small values capture more local structure.

# Keyword Arguments
- `searchctx = getcontext(model.index)`: the search context for the knn index (caches, hyperparameters, loggers, etc)
- `n_epochs::Integer = 30`: the number of training epochs for embedding optimization
- `learning_rate::Real = 1`: the initial learning rate during optimization
- `learning_rate_decay::Real = 0.8`: A decay factor for the `learning_rate` param (on each epoch)
- `set_operation_ratio::Real = 1`: interpolates between fuzzy set union and fuzzy set intersection when constructing the UMAP graph (global fuzzy simplicial set). The value of this parameter should be between 1.0 and 0.0: 1.0 indicates pure fuzzy union, while 0.0 indicates pure fuzzy intersection.
- `local_connectivity::Integer = 1`: the number of nearest neighbors that should be assumed to be locally connected. The higher this value, the more connected the manifold becomes. This should not be set higher than the intrinsic dimension of the manifold.
- `repulsion_strength::Real = 1`: the weighting of negative samples during the optimization process.
- `neg_sample_rate::Integer = 5`: the number of negative samples to select for each positive sample. Higher values will increase computational cost but result in slightly more accuracy.
- `tol::Real = 1e-4`: tolerance to early stopping while optimizing embeddings.
- `minbatch=0`: controls how parallel computation is made. See [`SimilaritySearch.getminbatch`](@ref) and `@batch` (`Polyester` package).
"""
function predict(model::UMAP,
                knns::AbstractMatrix{<:Integer},
                dists::AbstractMatrix{<:AbstractFloat};
                n_epochs::Integer = 30,
                learning_rate::Real = 1.0,
                learning_rate_decay::Real = 0.8,
                set_operation_ratio::Real = 1.0,
                local_connectivity::Integer = 1,
                repulsion_strength::Real = 1.0,
                neg_sample_rate::Integer = 5,
                tol::Real=1e-4,
                minbatch::Integer = 0
    )
    
    set_operation_ratio = convert(Float32, set_operation_ratio)
    learning_rate = convert(Float32, learning_rate)
    repulsion_strength = convert(Float32, repulsion_strength)
    learning_rate_decay = convert(Float32, learning_rate_decay)

    # argument checking
    learning_rate > 0                                || throw(ArgumentError("learning_rate must be greater than 0"))
    0 ≤ set_operation_ratio ≤ 1                      || throw(ArgumentError("set_operation_ratio must lie in [0, 1]"))
    local_connectivity > 0                           || throw(ArgumentError("local_connectivity must be greater than 0"))
    
    n_epochs = max(1, n_epochs)
    # main algorithm
    n = size(model.embedding, 2)
    graph = fuzzy_simplicial_set(knns, dists, n, local_connectivity, set_operation_ratio, false, false; minbatch)
    E = initialize_embedding(graph, model.embedding)
    optimize_embedding(graph, E, model.embedding, n_epochs, learning_rate, repulsion_strength, neg_sample_rate, model.a, model.b; tol, learning_rate_decay, minbatch)
end

function predict(model::UMAP, Q;
        k::Integer=16,
        searchctx = getcontext(model.index),
        minbatch = 0,
        kwargs...
    )
    model.index === nothing && throw(ArgumentError("this UMAP model doesn't support solving knn queries since `model.index == nothing` please use the alternative function that accepts `knns` and `dists` matrices"))
    Q = convert(AbstractDatabase, Q)
    0 < k <= length(Q) || throw(ArgumentError("number of neighbors must be in 0 < k <= number of points"))
    knns = searchbatch(model.index, searchctx, Q, k)
	knns_ = Matrix{UInt32}(undef, size(knns)...)
	dists_ = Matrix{Float32}(undef, size(knns)...)

	for i in CartesianIndices(knns)
		p = knns[i]
		knns_[i] = p.id
		dists_[i] = p.weight
	end
    predict(model, knns_, dists_; minbatch, kwargs...)
end

"""
    fuzzy_simplicial_set(knns, dists, n_points, local_connectivity, set_op_ratio, apply_fuzzy_combine=true; minbatch=0) -> membership_graph::SparseMatrixCSC, 

Construct the local fuzzy simplicial sets of each point represented by its distances
to its `k` nearest neighbors, stored in `knns` and `dists`, normalizing the distances
on the manifolds, and converting the metric space to a simplicial set.
`n_points` indicates the total number of points of the original data, while `knns` contains
indices of some subset of those points (ie some subset of 1:`n_points`). If `knns` represents
neighbors of the elements of some set with itself, then `knns` should have `n_points` number of
columns. Otherwise, these two values may be inequivalent.
If `apply_fuzzy_combine` is true, use intersections and unions to combine
fuzzy sets of neighbors (default true).

The returned graph will have size (`n_points`, size(knns, 2)).
"""
function fuzzy_simplicial_set(knns::AbstractMatrix,
                              dists::AbstractMatrix,
                              n_points::Integer,
                              local_connectivity,
                              set_operation_ratio,
                              fitting=true,
                              apply_fuzzy_combine=true;
                              minbatch=0)
    # @time σs, ρs = smooth_knn_dists(dists, k, local_connectivity)
    # @time rows, cols, vals = compute_membership_strengths(knns, dists, σs, ρs)
    rows, cols, vals = compute_membership_strengths(knns, dists, local_connectivity, fitting; minbatch)
    # transform uses n_points != size(knns, 2)
    fs_set = sparse(rows, cols, vals, n_points, size(knns, 2))

    if apply_fuzzy_combine
        dropzeros!(combine_fuzzy_sets(fs_set, convert(Float32, set_operation_ratio)))
    else
        dropzeros!(fs_set)
    end
end

"""
    smooth_knn_dists(dists, k, local_connectivity; <kwargs>) -> knn_dists, nn_dists

Compute the distances to the nearest neighbors for a continuous value `k`. Returns
the approximated distances to the kth nearest neighbor (`knn_dists`)
and the nearest neighbor (nn_dists) from each point.
"""
function smooth_knn_dists_vector(col::AbstractVector, k::Integer, local_connectivity::Integer; niter::Integer=64, bandwidth::Float32=1f0)
    local_connectivity = max(1, min(k, local_connectivity))
    ρ, sp = _find_first_non_zero(col, local_connectivity) #col[local_connectivity]
    σ = smooth_knn_dist_opt_binsearch((@view col[sp:end]), ρ, k-sp+1, bandwidth, niter)
    #σ = smooth_knn_dist_opt_binsearch((@view col[sp:end]), ρ, k, bandwidth, niter)
    ρ, σ
end

function _find_first_non_zero(v, sp)
    @inbounds for i in sp:length(v)
        v[i] > 0f0 && return v[i], i
    end
    
    v[1], 1
end

# calculate sigma for an individual point
function smooth_knn_dist_kernel(dists, ρ, mid)
    D::Float32 = 0.0
    invmid::Float32 = -1f0/mid
    @fastmath @inbounds @simd for d in dists
        d = d - ρ
        D += ifelse(d <= 0, 1f0, exp(d * invmid))
    end

    D
end

@fastmath function smooth_knn_dist_opt_binsearch(dists::AbstractVector, ρ, k, bandwidth, niter)
    target = bandwidth * log2(k)
    #target == 0 && return eps(Float32)  # a small number
    lo::Float32 = 0f0
    mid::Float32 = 1f0
    hi::Float32 = Inf32

    for _ in 1:niter
        psum = smooth_knn_dist_kernel(dists, ρ, mid)
        abs(psum - target) < SMOOTH_K_TOLERANCE && break
        if psum > target
            hi = mid
            mid = (lo + hi) * 0.5f0
        else
            lo = mid
            if hi === Inf32
                mid += mid
            else
                mid = (lo + hi) * 0.5f0
            end
        end
    end

    # TODO: set according to min k dist scale
    mid
end

"""
    compute_membership_strengths(knns, dists, local_connectivity, fitting; minbatch=0) -> rows, cols, vals

Compute the membership strengths for the 1-skeleton of each fuzzy simplicial set.
"""
function compute_membership_strengths(knns::AbstractMatrix, dists::AbstractMatrix, local_connectivity::Integer, fitting::Bool; minbatch=0)
    n = length(knns)
    rows = Vector{Int32}(undef, n)
    cols = Vector{Int32}(undef, n)
    vals = Vector{Float32}(undef, n)

    sizehint!(rows, n); sizehint!(cols, n); sizehint!(vals, n)
    n_neighbors, n = size(knns) # WARN n is now different
    minbatch = SimilaritySearch.getminbatch(minbatch, n)

    @batch minbatch=minbatch per=thread for i in 1:n
        D = @view dists[:, i]
        I = @view knns[:, i]
        ρ, σ = smooth_knn_dists_vector(D, n_neighbors, local_connectivity)
        invσ = -1f0 / σ
        ii = (i-1) * n_neighbors
        @inbounds for (k, objID) in enumerate(I)
            # D[k] <= ZERO_DISTANCE_THRESHOLD  # i == objID and near dups || objID == 0 (invalid objects)
            if fitting && i == objID || objID == 0
                d = 0f0
            else
                d = D[k] - ρ
                d = d <= 0 ? 1f0 : exp(d * invσ)
            end
            
            iii = ii + k
            cols[iii] = i
            rows[iii] = knns[k, i]
            vals[iii] = d
        end
    end
    
    rows, cols, vals
end
