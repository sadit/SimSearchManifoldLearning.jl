[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sadit.github.io/SimSearchManifoldLearning.jl/dev/)
[![Build Status](https://github.com/sadit/SimSearchManifoldLearning.jl/workflows/CI/badge.svg)](https://github.com/sadit/SimSearchManifoldLearning.jl/actions)

# SimilaritySearch and ManifoldLearning (and UMAP)
This package provides some support to use `SimilaritySearch` with manifold learning methods. In particular,
we implement the required methods to implement `knn` function for `ManifoldLearning` and also provides an `UMAP`
implementation that takes advantage of many `SimilaritySearch` features like multithreading and data independency; it supports string, sets, vectors, etc. under diverse distance functions.

The `ManifoldLearning` support is limited to some structure specification due to the design of the package. See the `ManifoldKnnIndex` type in the documentation pages.

# UMAP implementation

This package also provides a pure Julia implementation of the [Uniform Manifold Approximation and Projection](https://arxiv.org/abs/1802.03426) dimension reduction algorithm

> McInnes, L, Healy, J, Melville, J, *UMAP: Uniform Manifold Approximation and Projection for
> Dimension Reduction*. ArXiV 1802.03426, 2018

This implementation is based on the [UMAP.jl](https://github.com/dillondaudert/UMAP.jl) package by Dillon Gene Daudert and collaborators.

The provided implementation partially supports the `ManifoldLearning` API using `fit` and `predict` and similar arguments.
It can use any distance function from `SimilaritySearch`, [Distances.jl](https://github.com/JuliaStats/Distances.jl), [StringDistances.jl](https://github.com/matthieugomez/StringDistances.jl), or any distance function implemented by the user.

Additionally,  it improves multithreading support in the KNN and the UMAP projection.

## Examples and demonstrations

(https://sadit.github.io/SimilaritySearchDemos/)[https://sadit.github.io/SimilaritySearchDemos/] 

NOTE: Currently, they are working with direct implementations of this package. This will be changed soon (I will start once the package becomes part of the general registry).

## Disclaimer
This implementation is a work-in-progress. If you encounter any issues, please create
an issue or make a pull request.
