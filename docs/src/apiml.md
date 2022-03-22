```@meta

CurrentModule = SimSearchManifoldLearning
DocTestSetup = quote
    using SimSearchManifoldLearning
end
```

## SimilaritySearch and ManifoldLearning

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
