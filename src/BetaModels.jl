module BetaModels

using Reexport
@reexport using Distributions
using QuadGK
using StatsBase: counts

import Distributions: Normal
import Base: (-)

export ℙless, ℙbest


"""
    Normal(B::Beta)

Return `Normal` approximation to a `Beta`. 
No checks are made as to the adequacy of this approximation.
"""
function Distributions.Normal(B::Beta)
    a, b = params(B)
    adequate =
        isapprox((a + 1.0) / (a - 1.0), 1.0, atol = 1e-1) &
        isapprox((b + 1.0) / (b - 1.0), 1.0, atol = 1e-1)
    m = a / (a + b)
    v = a * b / ((a + b)^2 * (a + b + 1))
    if !adequate
        @warn "Normal approximation to `Beta($(a), $(b))` may be poor due to small parameter value."
    end
    return Normal(m, sqrt(v))
end


"""
    Base.:-(N::Normal{<:Real})

Return the negative of a `Normal` random variable.
"""
function Base.:-(N::Normal{<:Real})
    return Normal(-location(N), scale(N))
end


"""
    Base.:-(N1::Normal{<:Real}, N2::Normal{<:Real})
    
Subtract a normal random variable `N2` from another `N1`.
"""
function Base.:-(N1::Normal{<:Real}, N2::Normal{<:Real})
    return convolve(N1, -N2)
end


"""
    ℙless(D1::Beta, D2::Beta; δ = 0.0, method = "approx")

Probability that `D1 < D2 + δ`.
`method` may be one of 
`"approx"` (Normal approximation), 
`"montecarlo"` (Monte-Carlo approximation), or 
`"numeric"` (quadrature).
"""
function ℙless(D1::Beta, D2::Beta; δ = 0.0, method = "approx")
    if method == "approx"
        N1 = Normal(D1)
        N2 = Normal(D2)
        return cdf(N1 - N2, δ)
    elseif method == "montecarlo"
        return mean(rand(D1, 10_000) .< rand(D2, 10_000))
    elseif method == "numeric"
        f(x) = pdf(D1, x) * cdf(D2, x - δ)
        ∫, err = quadgk(f, δ, 1)
        if err > 1e-8
            @warn "Integration error $(err)"
        end
        return ∫
    end
end



"""
    ℙmax(D::Vector{Beta{Float64}}; n = 10_000)

Probability that each `Beta ∈ D` is maximum via Monte Carlo simulation.
"""
function ℙmax(D::Vector{Beta{Float64}}; n = 10_000, minimum = false)
    r = reduce(hcat, rand.(D, n))
    m = [x[2] for x in findmax(r, dims = 2)[2]]
    c = counts(m, 1:size(r, 2)) ./ size(r, 1)
    return c
end


"""
    ℙmin(D::Vector{Beta{Float64}}; n = 10_000)

Probability that each `Beta ∈ D` is maximum via Monte Carlo simulation.
"""
function ℙmin(D::Vector{Beta{Float64}}; n = 10_000, minimum = false)
    r = reduce(hcat, rand.(D, n))
    m = [x[2] for x in findmin(r, dims = 2)[2]]
    c = counts(m, 1:size(r, 2)) ./ size(r, 1)
    return c
end


end # end module