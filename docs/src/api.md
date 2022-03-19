```@meta

CurrentModule = SimSearchManifoldLearning
DocTestSetup = quote
    using SimSearchManifoldLearning
end
```

## UMAP 
```@docs
UMAP
fit
predict
optimize_embeddings!
```

### Layouts
```@docs
RandomLayout
SpectralLayout
PrecomputedLayout
KnnGraphLayout
```

### Precomputed Knn matrices
If you don't want to use `SimilaritySearch` for solving `k` nearest neighbors, you can also
pass precomputed knn solutions or distance matrices.

```@docs
PrecomputedKnns
PrecomputedAffinityMatrix
```

## SimilaritySearch and ManifoldLearning
## Distance functions

```@docs
ManifoldKnnIndex
```

The distance functions are defined to work under the `evaluate(::SemiMetric, u, v)` function (borrowed from [Distances.jl](https://github.com/JuliaStats/Distances.jl) package).


### KNN predefined types

```@docs
ExactEuclidean
ExactManhattan
ExactChebyshev
ExactCosine
ExactAngle
ApproxEuclidean
ApproxManhattan
ApproxChebyshev
ApproxCosine
ApproxAngle
```
