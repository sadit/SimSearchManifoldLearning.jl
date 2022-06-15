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
                            learning_rate_decay::Float32=0.9f0,
                            minbatch=0)
    self_reference = query_embedding_ === ref_embedding_  # is it training mode?
    self_reference && (query_embedding_ = copy(ref_embedding_))
    query_embedding = MatrixDatabase(query_embedding_)
    ref_embedding = MatrixDatabase(ref_embedding_)
    #learning_rate_step = convert(Float32, learning_rate / n_epochs + eps(typeof(learning_rate)))
    GR = rowvals(graph)
    # NZ = nonzeros(graph)

    minbatch = SimilaritySearch.getminbatch(minbatch, size(graph, 2))
    
    for _ in 1:n_epochs
        @batch minbatch=minbatch per=thread for i in 1:size(graph, 2)
            @inbounds QEi = query_embedding[i]
            @inbounds for ind in nzrange(graph, i)
                j = GR[ind]
                REj = ref_embedding[j]
                _gd_loop(QEi, REj, a, b, learning_rate)

                for _ in 1:neg_sample_rate
                    k = rand(eachindex(ref_embedding))
                    i == k && self_reference && continue
                    _gd_neg_loop(QEi, ref_embedding[k], a, b, repulsion_strength, learning_rate)
                end
            end
        end

        if self_reference # training -> update embedding
            ref_embedding_ .= query_embedding_
        end

        learning_rate = max(learning_rate * learning_rate_decay, 1f-4)
    end

    query_embedding_
end

@inline function _gd_loop(QEi, REj, a::Float32, b::Float32, learning_rate::Float32)
    sdist = evaluate(SqEuclidean(), QEi, REj)
    sdist < eps(Float32) && return
    delta = (-2f0 * a * b * sdist^(b-1f0))/(1f0 + a * sdist^b)

    @inbounds @simd for d in eachindex(QEi)
        grad = clamp(delta * (QEi[d] - REj[d]), -4f0, 4f0)
        #grad = delta * (QEi[d] - REj[d])
        QEi[d] += learning_rate * grad
    end
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