abstract type AbstractLayout end

export SpectralLayout, RandomLayout, PrecomputedLayout

struct SpectralLayout <: AbstractLayout end
struct RandomLayout <: AbstractLayout end
struct PrecomputedLayout <: AbstractLayout
    init::Matrix{Float32}
end

function initialize_embedding(layout::PrecomputedLayout, graph::AbstractMatrix, n_components)
    k, n = size(layout.init)
    if n_components !=  k || n != size(graph, 2)
        error("PrecomputedLayout matrix ($k, $n) doesn't matches ($n_components, $(size(graph, 2)))")
    end

    PrecomputedLayout.init
end

function initialize_embedding(::SpectralLayout, graph::AbstractMatrix{T}, n_components) where {T}
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
        embed = initialize_embedding(RandomLayout(), graph, n_components)
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

function initialize_embedding(::RandomLayout, graph::AbstractMatrix, n_components)
    randn(-10f0:eps(Float32):10f0, n_components, size(graph, 1))
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
