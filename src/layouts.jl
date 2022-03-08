abstract type AbstractLayout end

export SpectralLayout, RandomLayout, PrecomputedLayout, KnnGraphComponents

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

Initializes the embedding using a previously computed layout, i.e., (n_components, n_points) matrix. 
"""
struct PrecomputedLayout <: AbstractLayout
    init::Matrix{Float32}
end

"""
    KnnGraphComponents <: AbstractLayout


A lattice like + cloulds of points initialization that uses the computed all-knn graph
"""
struct KnnGraphComponents <: AbstractLayout
end

function initialize_embedding(::RandomLayout, graph::AbstractMatrix, knns, dists, n_components)
    n = size(knns, 2)
    rand(-10f0:eps(Float32):10f0, n_components, n)
end

function initialize_embedding(layout::PrecomputedLayout, graph::AbstractMatrix, knns, dists, n_components)
    k, n = size(layout.init)
    n_ = size(knns, 2)
    if n_components !=  k || n != n_
        error("PrecomputedLayout matrix ($k, $n) doesn't matches ($n_components, $(n_))")
    end

    layout.init
end

function initialize_embedding(::SpectralLayout, graph::AbstractMatrix{T}, knns, dists, n_components) where {T}
    local embed

    try
        embed = spectral_layout(graph, n_components)
        r = randn(T, size(embed))
        r .*= 0.0001
        # expand
        embed .*= 10 / maximum(embed)
        embed .+= r
    catch e
        @info "$e\nError encountered in spectral_layout; defaulting to random layout"
        embed = initialize_embedding(RandomLayout(), graph, knns, dists)
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

function initialize_embedding(layout::KnnGraphComponents, graph::AbstractMatrix, knns, dists, n_components)
    n = size(knns, 2)
    embed = zeros(Float32, n_components, n)
    E = MatrixDatabase(embed)
    L = MatrixDatabase(knns)
    C = zeros(Int32, n)

    for i in 1:n
        # C[i] != 0 && continue  ## multithreading
        prev = i
        links = L[i]
        while (curr = minimum(links); prev > curr)
            links = L[curr]
            prev = curr
        end

        #  C[prev] == 0 && rand_point_lattice_(E[i]) ## multithreading + also needs locks
        C[i] = prev
        if i == prev
            rand_point_lattice_(E[i])
        else
            rand_point_cloud_(E[i], E[prev], 0.25f0)
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
    size(V), size(center), min_dist
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
