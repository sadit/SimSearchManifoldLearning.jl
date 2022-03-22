```@meta
CurrentModule = SimSearchManifoldLearning
```

# Using SimilaritySearch for ManifoldLearning

One of the main component of non-linear projection methods is the solution of large batches of `k` nearest neighbors queries. In this sense [`SimilaritySearch`](https://github.com/sadit/SimilaritySearch.jl) provides fast multithreaded algorithms using exact and approximated searching algorihtms. 

This package provides two main structs `UMAP`, and `ManifoldKnnIndex` (to work with `ManifoldLearning`)

## UMAP
This package provides a pure Julia implementation of the [Uniform Manifold Approximation and Projection](https://arxiv.org/abs/1802.03426) dimension reduction algorithm

> McInnes, L, Healy, J, Melville, J, *UMAP: Uniform Manifold Approximation and Projection for
> Dimension Reduction*. ArXiV 1802.03426, 2018

The implementation in this package is based on the [UMAP.jl](https://github.com/dillondaudert/UMAP.jl) package by Dillon Gene Daudert and collaborators. Forked and adapted to work with `SimilaritySearch` and take advantage of multithreading systems and different layout initializations. It can use any distance function from `SimilaritySearch`, [Distances.jl](https://github.com/JuliaStats/Distances.jl), [StringDistances.jl](https://github.com/matthieugomez/StringDistances.jl), or any distance function implemented by the user.

`SimSearchManifoldLearning` provides an implementation that partially supports the `ManifoldLearning` API using `fit` and `predict`, and similar arguments.
## `ManifoldLearning` and `ManifoldKnnIndex`

We implemented the `k` nearest neighbor solver API of `ManifoldLearning` such that `SimilaritySearch` can be used. The use of `SimilaritySearch` gives multithreading `k` nearest neighbors solution, different distance functions and the different treadoffs between quality and speed for large datasets for the `k` nearest neighbors queries.

## Examples
Some examples are already working in [SimilaritySearchDemos](https://github.com/sadit/SimilaritySearchDemos).