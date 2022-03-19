using ManifoldLearning

export ManifoldKnnIndex,
    ExactEuclidean,
    ExactManhattan,
    ExactChebyshev,
    ExactCosine,
    ExactAngle,
    ApproxEuclidean,
    ApproxManhattan,
    ApproxChebyshev,
    ApproxCosine,
    ApproxAngle

"""
    ManifoldKnnIndex{IndexType,DistType,MinRecall}

Implements the `ManifoldLearning.AbstractNearestNeighbors` interface to interoperate with
the non-linear dimensionality reduction methods of the `ManifoldLearning` package.

It should be passed to the `fit` method as a type, e.g.,

```julia
fit(ManifoldKnnIndex{SearchGraph,L2Distance,0.9})
# or 
fit(ManifoldKnnIndex{ExhaustiveSearch,L1Distance,0})
```

IndexType should be some index of `SimilaritySearch` (ExhaustiveSearch or SearchGraph)

DistType should be anyone of the `SimilaritySearch` package or `Distances.jl`
or any other following the `SemiMetric`.

The third argument is only for `SearchGraph` and is applied to adjust the model
to achieve a desired recall score quality. It takes values from 0 to 1,
small values produce faster searches with lower qualities and high values
slower searches with higher quality. Values 0.8 or 0.9 may be desirable.

TODO NOTE: Use 0 to let ParetoRecall and 1 to specify ExhaustiveSearch.

"""
struct ManifoldKnnIndex{IndexType,DistType,MinRecall} <: ManifoldLearning.AbstractNearestNeighbors
    index::AbstractSearchContext  # A simple wrapper
end

"""
    ExactEuclidean

`ManifoldKnnIndex`'s type specialization for exact search with the Euclidean distance.
"""
const ExactEuclidean = ManifoldKnnIndex{ExhaustiveSearch,L2Distance,0}

"""
    ExactManhattan

`ManifoldKnnIndex`'s type specialization for exact search with the Manhattan distance.
"""
const ExactManhattan = ManifoldKnnIndex{ExhaustiveSearch,L1Distance,0}

"""
    ExactChebyshev

`ManifoldKnnIndex`'s type specialization for exact search with the Chebyshev distance.
"""
const ExactChebyshev = ManifoldKnnIndex{ExhaustiveSearch,LInftyDistance,0}

"""
    ExactCosine

`ManifoldKnnIndex`'s type specialization for exact search with the cosine distance.
"""
const ExactCosine = ManifoldKnnIndex{ExhaustiveSearch,CosineDistance,0}

"""
    ExactAngle

`ManifoldKnnIndex`'s type specialization for exact search with the angle distance.
"""
const ExactAngle = ManifoldKnnIndex{ExhaustiveSearch,AngleDistance,0}

"""
    ApproxEuclidean

`ManifoldKnnIndex`'s type specialization for approximate search with the Euclidean distance (expected recall of 0.9)
"""
const ApproxEuclidean = ManifoldKnnIndex{SearchGraph,L2Distance,0.9}

"""
    ApproxManhattan

`ManifoldKnnIndex`'s type specialization for approximate search with the Manhattan distance (expected recall of 0.9)
"""
const ApproxManhattan = ManifoldKnnIndex{SearchGraph,L1Distance,0.9}

"""
    ApproxChebyshev

`ManifoldKnnIndex`'s type specialization for approximate search with the Chebyshev distance (expected recall of 0.9)
"""
const ApproxChebyshev = ManifoldKnnIndex{SearchGraph,LInftyDistance,0.9}

"""
    ApproxCosine

`ManifoldKnnIndex`'s type specialization for approximate search with the Cosine distance (expected recall of 0.9)
"""
const ApproxCosine = ManifoldKnnIndex{SearchGraph,CosineDistance,0.9}

"""
    ApproxAngle

`ManifoldKnnIndex`'s type specialization for approximate search with the angle distance (expected recall of 0.9)
"""
const ApproxAngle = ManifoldKnnIndex{SearchGraph,AngleDistance,0.9}

Base.size(G::ManifoldKnnIndex) = (length(G.index[1]), length(G.index))

function fit(::Type{ManifoldKnnIndex{IndexType,DistType,MinRecall_}}, X) where {IndexType,DistType,MinRecall_}
    db = MatrixDatabase(X)
    dist = DistType()

    index = if IndexType === ExhaustiveSearch
        ExhaustiveSearch(; dist, db)
    else
        G = SearchGraph(; dist, db)
        parallel_block = length(db) < 512 || Threads.nthreads() == 1 ? 1 : 4 * Threads.nthreads()
        index!(G; parallel_block)
        minrecall = (MinRecall_ isa AbstractFloat) ? MinRecall_ : 0.9
        optimize!(G, OptimizeParameters(; kind=MinRecall(minrecall)))
        G
    end

    ManifoldKnnIndex{IndexType,DistType,MinRecall_}(index)
end
    
function ManifoldLearning.knn(G::ManifoldKnnIndex, Q::AbstractMatrix{T}, k::Integer; self::Bool=false, weights::Bool=true, kwargs...) where {T<:Real}
    m, n = size(Q)
    self && throw(ArgumentError("SimilaritySearch `"))
    n > k || throw(ArgumentError("Number of observations must be more than $(k)"))
    Q = MatrixDatabase(Q)
    KNNS = [KnnResult(k+self) for _ in 1:n]  # we can't use `allknn` efficiently from ManifoldLearning
    @time searchbatch(G.index, Q, KNNS; parallel=true)

    E = [res.id for res in KNNS] # reusing the internal structure
    W = [res.dist for res in KNNS] # `KnnResult` distances are always Float32

    if self  # removes zero/close-to-zero neighbors
        for i in eachindex(E)
            while length(W[i]) > 0
                if W[1] < eps(W)
                    popfirst!(E)
                    popfirst!(W)
                end
            end
        end
    end

    @show self, weights
    E, W
end
