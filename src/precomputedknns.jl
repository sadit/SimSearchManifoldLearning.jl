
export PrecomputedAffinityMatrix, PrecomputedKnns

"""
    struct PrecomputedAffinityMatrix <: AbstractSearchIndex
        dists # precomputed distances for all pairs (squared matrix)
    end

An index-like wrapper for precomputed affinity matrix.
"""
struct PrecomputedAffinityMatrix{MType<:AbstractMatrix} <: AbstractSearchIndex
    dists::MType
end

function SimilaritySearch.search(p::PrecomputedAffinityMatrix, q::Integer, res::KnnResult; pools=nothing)
    D = @view p.dists[:, q]
    @inbounds for i in eachindex(D)
        push_item!(res, i, D[i])
    end

    res
end

SimilaritySearch.getpools(::PrecomputedAffinityMatrix) = SimilaritySearch.GlobalKnnResult
SimilaritySearch.database(p::PrecomputedAffinityMatrix) = VectorDatabase(1:size(p.dists, 2))
SimilaritySearch.database(p::PrecomputedAffinityMatrix, i) = i

"""
    struct PrecomputedKnns <: AbstractSearchIndex
        knns
        dists
    end

An index-like wrapper for precomputed all-knns (as knns and dists matrices (k, n))
"""
struct PrecomputedKnns{KnnsType<:AbstractMatrix,DistsType<:AbstractMatrix,DBType<:AbstractVector} <: AbstractSearchIndex
    knns::KnnsType
    dists::DistsType
    db::DBType
end

function PrecomputedKnns(knns, dists)
    @assert size(knns) == size(dists)
    PrecomputedKnns(knns, dists, 1:size(knns, 2))
end

SimilaritySearch.getpools(::PrecomputedKnns) = SimilaritySearch.GlobalKnnResult
SimilaritySearch.database(p::PrecomputedKnns) = VectorDatabase(1:size(p.dists, 2))
SimilaritySearch.database(p::PrecomputedKnns, i) = i

function SimilaritySearch.search(p::PrecomputedKnns, q::Integer, res::KnnResult; pools=nothing)
    N = @view p.knns[:, q]
    D = @view p.dists[:, q]

    @inbounds for i in eachindex(N)
        push_item!(res, N[i], D[i])
    end

    res
end
