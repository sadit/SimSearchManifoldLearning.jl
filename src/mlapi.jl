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

struct ManifoldKnnIndex{IndexType,DistType,MinRecall} <: ManifoldLearning.AbstractNearestNeighbors
    index::AbstractSearchContext  # A simple wrapper
end

const ExactEuclidean = ManifoldKnnIndex{ExhaustiveSearch,L2Distance,0}
const ExactManhattan = ManifoldKnnIndex{ExhaustiveSearch,L1Distance,0}
const ExactChebyshev = ManifoldKnnIndex{ExhaustiveSearch,LInftyDistance,0}
const ExactCosine = ManifoldKnnIndex{ExhaustiveSearch,CosineDistance,0}
const ExactAngle = ManifoldKnnIndex{ExhaustiveSearch,AngleDistance,0}
const ApproxEuclidean = ManifoldKnnIndex{SearchGraph,L2Distance,0.9}
const ApproxManhattan = ManifoldKnnIndex{SearchGraph,L1Distance,0.9}
const ApproxChebyshev = ManifoldKnnIndex{SearchGraph,LInftyDistance,0.9}
const ApproxCosine = ManifoldKnnIndex{SearchGraph,CosineDistance,0.9}
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
