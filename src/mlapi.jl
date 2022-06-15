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
    ManifoldKnnIndex{DistType,MinRecall}

Implements the `ManifoldLearning.AbstractNearestNeighbors` interface to interoperate with
the non-linear dimensionality reduction methods of the `ManifoldLearning` package.

It should be passed to the `fit` method as a type, e.g.,

```julia
fit(ManifoldKnnIndex{L2Distance,0.9})  # will use an approximate index with an expected recall of 0.9
```

`DistType` should be any in [`SimilaritySearch`](https://sadit.github.io/SimilaritySearch.jl/dev/api/#Distance-functions) package, or `Distances.jl`
or any other following the `SemiMetric`.

The second argument of the composite type indicates the quality and therefore the type of index to use:
- It takes values between `0` and `1`.
- `0` means for a `SearchGraph` index using `ParetoRecall` optimization for the construction and searching, this will try to achieve a competitive structure in both quality and search speed
- `1` means for a `ExhaustiveSearch` index, this will compute the exact solution (exact knns) but at cost speed. Can work pretty well on small datasets and very high dimensionality. Really high dimensions suffer from the _curse of dimensionality_ such that an index like `SearchGraph` degrades to ExhaustiveSearch.
- `0 < value < 1`: Uses a `SearchGraph` and is the minimum recall-score quality that the index should perform. In particular, it constructs the index using `ParetoRecall` and the use a final optimization with `MinRecall`. It takes values from 0 to 1, small values produce faster searches with lower qualities and high values slower searches with higher quality. Values 0.8 or 0.9 should work pretty well.


Note: The minimum performance is evaluated in a small training set took from the database, this could yield to some kind of overfitting in the parameters, and therefore, perform not so good in an unseen query set. If you note this effect, please see `SimilaritySearch` documentation [function `optimize!`](https://sadit.github.io/SimilaritySearch.jl/dev/api/#SimilaritySearch.optimize!-Tuple{SearchGraph,%20OptimizeParameters}).

"""
struct ManifoldKnnIndex{DistType,MinRecall} <: ManifoldLearning.AbstractNearestNeighbors
    index::AbstractSearchContext  # A simple wrapper
end

"""
    ExactEuclidean

`ManifoldKnnIndex`'s type specialization for exact search with the Euclidean distance.
"""
const ExactEuclidean = ManifoldKnnIndex{L2Distance,1}

"""
    ExactManhattan

`ManifoldKnnIndex`'s type specialization for exact search with the Manhattan distance.
"""
const ExactManhattan = ManifoldKnnIndex{L1Distance,1}

"""
    ExactChebyshev

`ManifoldKnnIndex`'s type specialization for exact search with the Chebyshev distance.
"""
const ExactChebyshev = ManifoldKnnIndex{LInftyDistance,1}

"""
    ExactCosine

`ManifoldKnnIndex`'s type specialization for exact search with the cosine distance.
"""
const ExactCosine = ManifoldKnnIndex{CosineDistance,1}

"""
    ExactAngle

`ManifoldKnnIndex`'s type specialization for exact search with the angle distance.
"""
const ExactAngle = ManifoldKnnIndex{AngleDistance,1}

"""
    ApproxEuclidean

`ManifoldKnnIndex`'s type specialization for approximate search with the Euclidean distance (expected recall of 0.9)
"""
const ApproxEuclidean = ManifoldKnnIndex{L2Distance,0.9}

"""
    ApproxManhattan

`ManifoldKnnIndex`'s type specialization for approximate search with the Manhattan distance (expected recall of 0.9)
"""
const ApproxManhattan = ManifoldKnnIndex{L1Distance,0.9}

"""
    ApproxChebyshev

`ManifoldKnnIndex`'s type specialization for approximate search with the Chebyshev distance (expected recall of 0.9)
"""
const ApproxChebyshev = ManifoldKnnIndex{LInftyDistance,0.9}

"""
    ApproxCosine

`ManifoldKnnIndex`'s type specialization for approximate search with the Cosine distance (expected recall of 0.9)
"""
const ApproxCosine = ManifoldKnnIndex{CosineDistance,0.9}

"""
    ApproxAngle

`ManifoldKnnIndex`'s type specialization for approximate search with the angle distance (expected recall of 0.9)
"""
const ApproxAngle = ManifoldKnnIndex{AngleDistance,0.9}


## ManifoldLearning api for nearest neighbor algorithms

Base.size(G::ManifoldKnnIndex) = (length(G.index[1]), length(G.index))

function fit(::Type{ManifoldKnnIndex{DistType,MinRecall_}}, X) where {DistType,MinRecall_}
    db = MatrixDatabase(X)
    dist = DistType()

    index = if MinRecall_ == 1
        ExhaustiveSearch(; dist, db)
    else
        G = SearchGraph(; dist, db)
        parallel_block = length(db) < 512 || Threads.nthreads() == 1 ? 1 : 4 * Threads.nthreads()
        index!(G; parallel_block)
        if MinRecall_ > 0
            minrecall = (MinRecall_ isa AbstractFloat) ? MinRecall_ : 0.9
            optimize!(G, OptimizeParameters(; kind=MinRecall(minrecall)))
        end
        G
    end

    ManifoldKnnIndex{DistType,MinRecall_}(index)
end

"""
    knn(G::ManifoldKnnIndex, Q::AbstractMatrix{T}, k::Integer; self::Bool=false, weights::Bool=true, kwargs...) where {T<:Real}

Solves `k` nearest neighbors queries of `Q` using `G` (a SimilaritySearch index).
"""
function ManifoldLearning.knn(G::ManifoldKnnIndex, Q::AbstractMatrix{T}, k::Integer; self::Bool=false, weights::Bool=true, minbatch=0, kwargs...) where {T<:Real}
    m, n = size(Q)
    self && throw(ArgumentError("SimilaritySearch `"))
    n > k || throw(ArgumentError("Number of observations must be more than $(k)"))
    Q = MatrixDatabase(Q)
    KNNS = [KnnResult(k+self) for _ in 1:n]  # we can't use `allknn` efficiently from ManifoldLearning
    @time searchbatch(G.index, Q, KNNS; minbatch)

    E = [idview(res) for res in KNNS] # reusing the internal structure
    W = [distview(res) for res in KNNS] # `KnnResult` distances are always Float32

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

    E, W
end
