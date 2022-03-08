module UMAP

using Arpack
using Distances
using LinearAlgebra
using LsqFit: curve_fit
using SimilaritySearch
using SparseArrays

include("utils.jl")
include("precomputedknns.jl")
include("layouts.jl")
include("embeddings.jl")
include("umap_.jl")

export umap, UMAP_, transform

end # module
