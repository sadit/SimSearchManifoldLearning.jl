
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
        push!(res, i, D[i])
    end

    res
end

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

function SimilaritySearch.search(p::PrecomputedKnns, q::Integer, res::KnnResult; pools=nothing)
    N = @view p.knns[:, q]
    D = @view p.dists[:, q]

    @inbounds for i in eachindex(N)
        push!(res, N[i], D[i])
    end

    res
end