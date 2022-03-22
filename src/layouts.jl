abstract type AbstractLayout end

export SpectralLayout, RandomLayout, PrecomputedLayout, KnnGraphLayout

"""
    SpectralLayout <: AbstractLayout

Initializes the embedding using the spectral layout method. It could be costly in very large datasets.
"""
struct SpectralLayout <: AbstractLayout end

"""
    RandomLayout <: AbstractLayout

Initializes the embedding using a random set of points. It may converge slowly
"""
struct RandomLayout <: AbstractLayout end

"""
    PrecomputedLayout <: AbstractLayout

Initializes the embedding using a previously computed layout, i.e., (maxoutdim, n_points) matrix. 
"""
struct PrecomputedLayout <: AbstractLayout
    init::Matrix{Float32}
end

"""
    KnnGraphLayout <: AbstractLayout


A lattice like + clouds of points initialization that uses the computed all-knn graph.
This layout initialization is a simple proof of concept, so please use it under this assumption.
"""
struct KnnGraphLayout <: AbstractLayout
end

function initialize_embedding(::RandomLayout, graph::AbstractMatrix, knns, dists, maxoutdim::Integer)
    n = size(knns, 2)
    rand(-10f0:eps(Float32):10f0, maxoutdim, n)
end

function initialize_embedding(layout::PrecomputedLayout, graph::AbstractMatrix, knns, dists, maxoutdim::Integer)
    k, n = size(layout.init)
    n_ = size(knns, 2)
    if maxoutdim !=  k || n != n_
        error("PrecomputedLayout matrix ($k, $n) doesn't matches ($maxoutdim, $(n_))")
    end

    layout.init
end

function initialize_embedding(::SpectralLayout, graph::AbstractMatrix{T}, knns, dists, maxoutdim::Integer) where {T}
    local embed

    try
        embed = spectral_layout(graph, maxoutdim)
        r = randn(T, size(embed))
        r .*= 0.0001
        # expand
        embed .*= 10 / maximum(embed)
        embed .+= r
    catch e
        @info "$e\nError encountered in spectral_layout; defaulting to random layout"
        embed = initialize_embedding(RandomLayout(), graph, knns, dists, maxoutdim)
    end

    embed
end

"""
    spectral_layout(graph, embed_dim) -> embedding

Initialize the graph layout with spectral embedding.
"""
function spectral_layout(graph::SparseMatrixCSC{T},
                         embed_dim::Integer) where {T<:Real}
    graph_f64 = convert.(Float64, graph)
    D_ = Diagonal(dropdims(sum(graph_f64; dims=2); dims=2))
    D = inv(sqrt(D_))
    # normalized laplacian
    L = Symmetric(I - D*graph*D)

    k = embed_dim+1
    num_lanczos_vectors = max(2k+1, round(Int, sqrt(size(L, 1))))
    # get the 2nd - embed_dim+1th smallest eigenvectors
    eigenvals, eigenvecs = eigs(L; nev=k,
                                   ncv=num_lanczos_vectors,
                                   which=:SM,
                                   tol=1e-4,
                                   v0=ones(Float64, size(L, 1)),
                                   maxiter=size(L, 1)*5)
    layout = permutedims(eigenvecs[:, 2:k])::Array{Float64, 2}
    return convert.(T, layout)
end

function revknn_frequencies_(knns)
    n = size(knns, 2)
    F = zeros(Int32, n)
    for i in 1:n
        for objID in view(knns, :, i)
            F[objID] += 1  # appoximation to indegree centrality
        end
    end

    F
end

function knn_fast_components_(knns)  ## approximated connected components
    C = Dict{Int32,Vector{Int32}}()
    for i in 1:size(knns, 2)
        prev = i
        links = @view knns[:, i]

        while (curr = minimum(links); prev > curr)
            links = @view knns[:, curr]
            prev = curr
        end

        lst = get(C, prev, nothing)
        if lst === nothing
            C[prev] = Int32[i]
        else
            push!(lst, i)
        end
    end

    C
end

function initialize_embedding(::KnnGraphLayout, graph::AbstractMatrix, knns, dists, maxoutdim::Integer)
    n = size(knns, 2)
    embed = zeros(Float32, maxoutdim, n)
    E = MatrixDatabase(embed)
    C = knn_fast_components_(knns)
    F = revknn_frequencies_(knns)

    for lst in values(C)
        _, p = findmax(p -> F[p], lst)
        i = lst[p]
        rand_point_lattice_(E[i])  # most popular elements are in the center
        for p in lst
            if p != i
                rand_point_cloud_(E[p], E[i], 0.25f0)
            end
        end
    end
    
    embed
end

function rand_point_lattice_(V)
    @inbounds for i in eachindex(V)
        V[i] = rand(-10f0:0.5f0:10f0)
    end
end

function rand_point_cloud_(V, center, min_dist::Float32)
    @inbounds for i in eachindex(V)
        V[i] = center[i] + rand(-min_dist:eps(Float32):min_dist)
    end
end


"""
    initialize_embedding(graph::AbstractMatrix, ref_embedding::Matrix) -> embedding

Initialize an embedding of points corresponding to the columns of the `graph`, by taking weighted means of
the columns of `ref_embedding`, where weights are values from the rows of the `graph`.

The resulting embedding will have shape `(size(ref_embedding, 1), size(graph, 2))`, where `size(ref_embedding, 1)`
is the number of components (dimensions) of the `reference embedding`, and `size(graph, 2)` is the number of 
samples in the resulting embedding. Its elements will have type T.
"""
function initialize_embedding(graph::AbstractMatrix, ref_embedding::Matrix{Float32})
    (ref_embedding * graph) ./ (sum(graph, dims=1) .+ eps(Float32))
end
