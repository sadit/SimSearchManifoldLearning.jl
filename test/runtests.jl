using Test
using Distances: Euclidean, CosineDist
using Random
using SparseArrays
using LinearAlgebra
using SimilaritySearch
using UMAP
using UMAP: initialize_embedding, fuzzy_simplicial_set, compute_membership_strengths, smooth_knn_dist_opt_binsearch, spectral_layout, optimize_embedding, combine_fuzzy_sets, fit_ab, SMOOTH_K_TOLERANCE


include("utils_tests.jl")
include("umap_tests.jl")
