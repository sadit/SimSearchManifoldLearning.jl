#=
Utilities used by UMAP.jl
=#


@inline fit_ab(min_dist, spread) = fit_ab(min_dist, spread, nothing, nothing)
@inline fit_ab(_, __, a, b) = a, b

"""
    fit_ab(min_dist, spread, _a, _b) -> a, b

Find a smooth approximation to the membership function of points embedded in ℜᵈ.
This fits a smooth curve that approximates an exponential decay offset by `min_dist`,
returning the parameters `(a, b)`.
"""
function fit_ab(min_dist, spread, ::Nothing, ::Nothing)::Tuple{Float32,Float32}
    ψ(d) = d >= min_dist ? exp(-(d - min_dist)/spread) : 1.
    xs = LinRange(0., spread*3, 300)
    ys = map(ψ, xs)
    @. curve(x, p) = (1. + p[1]*x^(2*p[2]))^(-1)
    result = curve_fit(curve, xs, ys, [1., 1.], lower=[0., -Inf])
    a, b = result.param
    return a, b
end


# combine local fuzzy simplicial sets
@inline function combine_fuzzy_sets(fs_set::AbstractMatrix,
                                    set_op_ratio::Float32)
    return set_op_ratio .* fuzzy_set_union(fs_set) .+
           (1f0 - set_op_ratio) .* fuzzy_set_intersection(fs_set)
end

@inline function fuzzy_set_union(fs_set::AbstractMatrix)
    return fs_set .+ fs_set' .- (fs_set .* fs_set')
end

@inline function fuzzy_set_intersection(fs_set::AbstractMatrix)
    return fs_set .* fs_set'
end
