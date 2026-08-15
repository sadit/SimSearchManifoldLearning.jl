# This file is part of SimSearchManifoldLearning.jl

"""
    PrecomputedKnns(knns::AbstractMatrix, dists::AbstractMatrix, db=1:size(knns, 2))

Search index based on precomputed nearest neighbors and distances matrices.
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

SimilaritySearch.database(p::PrecomputedKnns) = VectorDatabase(p.db)
SimilaritySearch.database(p::PrecomputedKnns, i) = p.db[i]

function SimilaritySearch.search(p::PrecomputedKnns, ctx::AbstractContext, q::Integer, res::AbstractKnnQueue)
    N = @view p.knns[:, q]
    D = @view p.dists[:, q]

    @inbounds for i in eachindex(N)
        push_item!(res, N[i], D[i])
    end

    res
end
