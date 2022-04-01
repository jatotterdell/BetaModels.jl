module BetaModels

using Reexport
@reexport using Distributions
using QuadGK
using StatsBase: counts

import Distributions: Normal
import Base: (-)

export ℙless, ℙbest


@doc raw"""
    Normal(B::Beta)

Return `Normal` approximation to a `Beta`. 

If ``B\sim\text{Beta}(a,b)`` then 
```math
B\approx\text{Normal}\left(\frac{a}{a+b}, \frac{ab}{(a+b)^2(a+b+1)}\right)
```
The approximation improves as ``a\approx b`` and for large values of ``a`` and ``b``.

An improvement can be obtained by transforming the parameter, [Wise's transformation](https://www.jstor.org/stable/2332968).
"""
function Distributions.Normal(B::Beta{Float64})
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


@doc raw"""
    Base.:-(X::Normal{<:Real})

Return the negative of a `Normal` random variable, i.e. ``X \sim \text{Normal}(\mu,\sigma^2)``
then return ``-X \sim \text{Normal}(-\mu,\sigma^2)``.
"""
function Base.:-(X::Normal{<:Real})
    return Normal(-location(X), scale(X))
end


@doc raw"""
    Base.:-(X::Normal{<:Real}, Y::Normal{<:Real})
    
Subtract a normal random variable `Y` from another normal random variable `X` according to
```math
\text{Normal}(\mu, \sigma^2) - \text{Normal}(\nu, \omega^2) \sim \text{Normal}(\mu-\nu, \sigma^2+\omega^2)
```
"""
function Base.:-(X::Normal{<:Real}, Y::Normal{<:Real})
    return convolve(X, -Y)
end


"""
    ℙless(X::Beta{Float64}, Y::Beta{Float64}; δ = 0.0, method = "approx")

Return `ℙ(X < Y + δ)`.
`method` may be one of 
`"approx"` (Normal approximation), 
`"montecarlo"` (Monte-Carlo approximation), or 
`"numeric"` (quadrature).
"""
function ℙless(X::Beta{Float64}, Y::Beta{Float64}; δ = 0.0, method = "approx")
    if method == "approx"
        N1 = Normal(X)
        N2 = Normal(Y)
        return cdf(N1 - N2, δ)
    elseif method == "montecarlo"
        return mean(rand(X, 10_000) .< rand(Y, 10_000))
    elseif method == "numeric"
        f(x) = pdf(X, x) * cdf(Y, x - δ)
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
function ℙmax(D::Vector{Beta{Float64}}; n = 10_000)
    r = reduce(hcat, rand.(D, n))
    m = [x[2] for x in findmax(r, dims = 2)[2]]
    c = counts(m, 1:size(r, 2)) ./ size(r, 1)
    return c
end


"""
    ℙmin(D::Vector{Beta{Float64}}; n = 10_000)

Probability that each `Beta ∈ D` is minimum via Monte Carlo simulation.
"""
function ℙmin(D::Vector{Beta{Float64}}; n = 10_000)
    r = reduce(hcat, rand.(D, n))
    m = [x[2] for x in findmin(r, dims = 2)[2]]
    c = counts(m, 1:size(r, 2)) ./ size(r, 1)
    return c
end


end # end module