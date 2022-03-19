var documenterSearchIndex = {"docs":
[{"location":"api/","page":"API","title":"API","text":"\nCurrentModule = SimSearchManifoldLearning\nDocTestSetup = quote\n    using SimSearchManifoldLearning\nend","category":"page"},{"location":"api/#UMAP","page":"API","title":"UMAP","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"UMAP\nfit\npredict\noptimize_embeddings!","category":"page"},{"location":"api/#StatsAPI.fit","page":"API","title":"StatsAPI.fit","text":"fit(Type{UMAP}, knns, dists [, maxoutdim=2]; <kwargs>) -> UMAP object\n\nCreate a model representing the embedding of data (X, dist) into maxoutdim-dimensional space. Note that knns and dists jointly specify the all k nearest neighbors of (X dist), these results must not include self-references. See the allknn method in SimilaritySearch.\n\nArguments\n\nknns: A (k n) matrix of integers (identifiers).\ndists: A (k n) matrix of floating points (distances).\n\nIt uses all available threads for the projection.\n\nKeyword Arguments\n\nmaxoutdim::Integer=2: The number of components in the embedding\nn_epochs::Integer = 300: the number of training epochs for embedding optimization\nlearning_rate::Real = 1: the initial learning rate during optimization\nlearning_rate_decay::Real = 0.9: how much learning_rate is updated on each epoch (learning_rate *= learning_rate_decay) (a minimum value is also considered as 1e-6)\nlayout::AbstractLayout = SpectralLayout(): how to initialize the output embedding\nmin_dist::Real = 0.1: the minimum spacing of points in the output embedding\nspread::Real = 1: the effective scale of embedded points. Determines how clustered embedded points are in combination with min_dist.\nset_operation_ratio::Real = 1: interpolates between fuzzy set union and fuzzy set intersection when constructing the UMAP graph (global fuzzy simplicial set). The value of this parameter should be between 1.0 and 0.0: 1.0 indicates pure fuzzy union, while 0.0 indicates pure fuzzy intersection.\nlocal_connectivity::Integer = 1: the number of nearest neighbors that should be assumed to be locally connected. The higher this value, the more connected the manifold becomes. This should not be set higher than the intrinsic dimension of the manifold.\nrepulsion_strength::Real = 1: the weighting of negative samples during the optimization process.\nneg_sample_rate::Integer = 5: the number of negative samples to select for each positive sample. Higher values will increase computational cost but result in slightly more accuracy.\na = nothing: this controls the embedding. By default, this is determined automatically by min_dist and spread.\nb = nothing: this controls the embedding. By default, this is determined automatically by min_dist and spread.\n\n\n\n\n\nfit(::Type{<:UMAP}, index_or_data;\n    k=15,\n    dist::SemiMetric=L2Distance,\n    kwargs...)\n\nWrapper for fit that computes n_nearests nearest neighbors on index_or_data and passes these and kwargs to regular fit.\n\nArguments\n\nindex_or_data: an already constructed index (see SimilaritySearch), a matrix, or an abstact database (SimilaritySearch)\nk=15: number of neighbors to compute\ndist=L2Distance(): A distance function (see Distances.jl)\n\n\n\n\n\nfit(UMAP::UMAP, maxoutdim; <kwargs>)\n\nReuses a previously computed model with a different number of components\n\nKeyword arguments\n\nn_epochs=50: number of epochs to run\nlearning_rate::Real = 1f0: initial learning rate\nlearning_rate_decay::Real = 0.9f0: how learning rate is adjusted per epoch learning_rate *= learning_rate_decay\nrepulsion_strength::Float32 = 1f0: repulsion force (for negative sampling)\nneg_sample_rate::Integer = 5: how many negative examples per object are used.\n\n\n\n\n\n","category":"function"},{"location":"api/#StatsAPI.predict","page":"API","title":"StatsAPI.predict","text":"predict(model::UMAP)\n\nReturns the internal embedding (the entire dataset projection)\n\n\n\n\n\npredict(model::UMAP, Q::AbstractDatabase; k::Integer=15, kwargs...)\npredict(model::UMAP, knns, dists; <kwargs>) -> embedding\n\nUse the given model to embed new points Q into an existing embedding produced by (X dist). The second function represent Q using its k nearest neighbors in X under some distance function (knns and dists) See searchbatch in SimilaritySearch to compute both (also for AbstractDatabase objects).\n\nArguments\n\nmodel: The fitted model\nknns: matrix of identifiers (integers) of size (k Q)\ndists: matrix of distances (floating point values) of size (k Q)\n\nNote: the number of neighbors k (embedded into knn matrices) control the embedding. Larger values capture more global structure in the data, while small values capture more local structure.\n\nKeyword Arguments\n\nn_epochs::Integer = 30: the number of training epochs for embedding optimization\nlearning_rate::Real = 1: the initial learning rate during optimization\nlearning_rate_decay::Real = 0.8: A decay factor for the learning_rate param (on each epoch)\nset_operation_ratio::Real = 1: interpolates between fuzzy set union and fuzzy set intersection when constructing the UMAP graph (global fuzzy simplicial set). The value of this parameter should be between 1.0 and 0.0: 1.0 indicates pure fuzzy union, while 0.0 indicates pure fuzzy intersection.\nlocal_connectivity::Integer = 1: the number of nearest neighbors that should be assumed to be locally connected. The higher this value, the more connected the manifold becomes. This should not be set higher than the intrinsic dimension of the manifold.\nrepulsion_strength::Real = 1: the weighting of negative samples during the optimization process.\nneg_sample_rate::Integer = 5: the number of negative samples to select for each positive sample. Higher values will increase computational cost but result in slightly more accuracy.\n\n\n\n\n\n","category":"function"},{"location":"api/#Layouts","page":"API","title":"Layouts","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"RandomLayout\nSpectralLayout\nPrecomputedLayout\nKnnGraphLayout","category":"page"},{"location":"api/#SimSearchManifoldLearning.RandomLayout","page":"API","title":"SimSearchManifoldLearning.RandomLayout","text":"RandomLayout <: AbstractLayout\n\nInitializes the embedding using a random set of points. It may converge slowly\n\n\n\n\n\n","category":"type"},{"location":"api/#SimSearchManifoldLearning.SpectralLayout","page":"API","title":"SimSearchManifoldLearning.SpectralLayout","text":"SpectralLayout <: AbstractLayout\n\nInitializes the embedding using the spectral layout method. It could be costly in very large datasets.\n\n\n\n\n\n","category":"type"},{"location":"api/#SimSearchManifoldLearning.PrecomputedLayout","page":"API","title":"SimSearchManifoldLearning.PrecomputedLayout","text":"PrecomputedLayout <: AbstractLayout\n\nInitializes the embedding using a previously computed layout, i.e., (maxoutdim, n_points) matrix. \n\n\n\n\n\n","category":"type"},{"location":"api/#SimSearchManifoldLearning.KnnGraphLayout","page":"API","title":"SimSearchManifoldLearning.KnnGraphLayout","text":"KnnGraphLayout <: AbstractLayout\n\nA lattice like + clouds of points initialization that uses the computed all-knn graph\n\n\n\n\n\n","category":"type"},{"location":"api/#Precomputed-Knn-matrices","page":"API","title":"Precomputed Knn matrices","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"If you don't want to use SimilaritySearch for solving k nearest neighbors, you can also pass precomputed knn solutions or distance matrices.","category":"page"},{"location":"api/","page":"API","title":"API","text":"PrecomputedKnns\nPrecomputedAffinityMatrix","category":"page"},{"location":"api/#SimSearchManifoldLearning.PrecomputedKnns","page":"API","title":"SimSearchManifoldLearning.PrecomputedKnns","text":"struct PrecomputedKnns <: AbstractSearchContext\n    knns\n    dists\nend\n\nAn index-like wrapper for precomputed all-knns (as knns and dists matrices (k, n))\n\n\n\n\n\n","category":"type"},{"location":"api/#SimSearchManifoldLearning.PrecomputedAffinityMatrix","page":"API","title":"SimSearchManifoldLearning.PrecomputedAffinityMatrix","text":"struct PrecomputedAffinityMatrix <: AbstractSearchContext\n    dists # precomputed distances for all pairs (squared matrix)\nend\n\nAn index-like wrapper for precomputed affinity matrix.\n\n\n\n\n\n","category":"type"},{"location":"api/#SimilaritySearch-and-ManifoldLearning","page":"API","title":"SimilaritySearch and ManifoldLearning","text":"","category":"section"},{"location":"api/#Distance-functions","page":"API","title":"Distance functions","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"ManifoldKnnIndex","category":"page"},{"location":"api/","page":"API","title":"API","text":"The distance functions are defined to work under the evaluate(::SemiMetric, u, v) function (borrowed from Distances.jl package).","category":"page"},{"location":"api/#KNN-predefined-types","page":"API","title":"KNN predefined types","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"ExactEuclidean\nExactManhattan\nExactChebyshev\nExactCosine\nExactAngle\nApproxEuclidean\nApproxManhattan\nApproxChebyshev\nApproxCosine\nApproxAngle","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = SimSearchManifoldLearning","category":"page"},{"location":"#Using-SimilaritySearch-for-ManifoldLearning","page":"Home","title":"Using SimilaritySearch for ManifoldLearning","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"SimilaritySearch.jl is a library for nearest neighbor search. In particular, it contains the implementation for SearchGraph, a fast and flexible search index.","category":"page"},{"location":"","page":"Home","title":"Home","text":"One of the main component of non-linear projection methods is the solution of large batches of k nearest neighbors queries. In this sense SimilaritySearch provides fast multithreaded algorithms using exact and approximated searching algorihtms.  ... work in progress ...","category":"page"},{"location":"#Notes:","page":"Home","title":"Notes:","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"... work in progress ...","category":"page"},{"location":"#UMAP-implementation","page":"Home","title":"UMAP implementation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"... work in progress ...","category":"page"},{"location":"#Examples:","page":"Home","title":"Examples:","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"... work in progress ...","category":"page"},{"location":"","page":"Home","title":"Home","text":"Some examples are already working in SimilaritySearchDemos","category":"page"}]
}