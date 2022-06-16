# initializing and optimizing embeddings

"""
    optimize_embedding(graph, query_embedding_, ref_embedding_, n_epochs, alpha, min_dist, spread, repulsion_strength, neg_sample_rate, _a=nothing, _b=nothing; minbatch=0) -> embedding

Optimize an embedding by minimizing the fuzzy set cross entropy between the high and low dimensional simplicial sets using stochastic gradient descent.
Optimize "query" samples with respect to "reference" samples. The optimization uses all available threads.

# Arguments
- `graph`: a sparse matrix of shape (n_samples, n_samples)
- `query_embedding_`: a vector of length (n_samples,) of vectors representing the embedded data points to be optimized ("query" samples)
- `ref_embedding_`: a vector of length (n_samples,) of vectors representing the embedded data points to optimize against ("reference" samples)
- `n_epochs`: the number of training epochs for optimization
- `learning_rate`: the initial learning rate
- `repulsion_strength`: the repulsive strength of negative samples
- `neg_sample_rate`: the number of negative samples per positive sample
- `_a`: this controls the embedding. If the actual argument is `nothing`, this is determined automatically by `min_dist` and `spread`.
- `_b`: this controls the embedding. If the actual argument is `nothing`, this is determined automatically by `min_dist` and `spread`.

# Keyword arguments:
- `tol=1e-4`: tolerance to early stopping optimization, smaller values could improve embeddings but use higher computational resources.
- `learning_rate_decay=0.9f0`: a scale factor for the learning rate (applied at each epoch)
- `minbatch=0`: controls how parallel computation is made. See [`SimilaritySearch.getminbatch`](@ref) and `@batch` (`Polyester` package).
"""
function optimize_embedding(graph,
                            query_embedding_::AbstractMatrix,
                            ref_embedding_::AbstractMatrix,
                            n_epochs::Int,
                            learning_rate::Float32,
                            repulsion_strength::Float32,
                            neg_sample_rate::Int,
                            a::Float32,
                            b::Float32;
                            tol::Real=1e-4,
                            learning_rate_decay::Float32=0.9f0,
                            minbatch=0)
    self_reference = query_embedding_ === ref_embedding_  # is it training mode?
    self_reference && (query_embedding_ = copy(ref_embedding_))
    query_embedding = MatrixDatabase(query_embedding_)
    ref_embedding = MatrixDatabase(ref_embedding_)
    GR = rowvals(graph)
    # NZ = nonzeros(graph)

    minbatch = SimilaritySearch.getminbatch(minbatch, size(graph, 2))
    
    prev = typemax(Float64)
    curr = Ref(0.0) 
    errlock = Threads.SpinLock()
    tol = convert(Float64, tol)
    errweight = 1 / length(query_embedding_)

    for ep in 1:n_epochs
        curr[] = 0.0

        @batch minbatch=minbatch per=thread for i in 1:size(graph, 2)
            err = 0.0 
            @inbounds QEi = query_embedding[i]
            range_ = nzrange(graph, i)
            @inbounds for ind in range_
                j = GR[ind]
                REj = ref_embedding[j]
                err += errweight * _gd_loop(QEi, REj, a, b, learning_rate)

                for _ in 1:neg_sample_rate
                    k = rand(eachindex(ref_embedding))
                    i == k && self_reference && continue
                    _gd_neg_loop(QEi, ref_embedding[k], a, b, repulsion_strength, learning_rate)
                end
            end

            lock(errlock)
            try
                curr[] += err / length(range_)
            finally
                unlock(errlock)
            end
        end

        @info ep => abs(prev - curr[]) => curr[]
        abs(prev - curr[]) < tol && begin     
            @info " --------------------------- ***** early stopping by convergence ***** -------------------------- "
            break
        end

        prev = curr[]
        if self_reference # training -> update embedding
            ref_embedding_ .= query_embedding_
        end

        learning_rate = max(learning_rate * learning_rate_decay, 1f-3)
    end

    query_embedding_
end

@inline function _gd_loop(QEi, REj, a::Float32, b::Float32, learning_rate::Float32)
    sdist = evaluate(SqEuclidean(), QEi, REj)
    sdist < eps(Float32) && return sdist
    delta = (-2f0 * a * b * sdist^(b-1f0))/(1f0 + a * sdist^b)

    @inbounds @simd for d in eachindex(QEi)
        grad = clamp(delta * (QEi[d] - REj[d]), -4f0, 4f0)
        QEi[d] += learning_rate * grad
    end

    sdist
end

@inline function _gd_neg_loop(QEi, REk, a::Float32, b::Float32, repulsion_strength::Float32, learning_rate::Float32)
    sdist = evaluate(SqEuclidean(), QEi, REk)
    if sdist > 0
        delta = (2f0 * repulsion_strength * b) / ((0.001f0 + sdist)*(1f0 + a * sdist^b))

        @inbounds @simd for d in eachindex(QEi)
            grad = clamp(delta * (QEi[d] - REk[d]), -4f0, 4f0)
            #grad = delta * (QEi[d] - REk[d])
            QEi[d] += learning_rate * grad
        end
    else
        @inbounds @simd for d in eachindex(QEi)
            QEi[d] += learning_rate * 4f0
        end
    end
end