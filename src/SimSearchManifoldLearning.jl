module SimSearchManifoldLearning

using Arpack
using Distances
using LinearAlgebra
using LsqFit: curve_fit
using SimilaritySearch
using SparseArrays
using Polyester

import StatsAPI: fit, predict
export fit, predict

include("utils.jl")
include("precomputedknns.jl")
include("layouts.jl")
include("embeddings.jl")
include("umap_.jl")
include("mlapi.jl")


end # module
