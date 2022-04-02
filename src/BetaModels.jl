module BetaModels

using Reexport
@reexport using Distributions
using QuadGK
using Parameters
using StatsBase: counts

import Distributions: Normal
import Base: (-)

export ℙless, ℙmax, ℙmin, Pℙless, PPoS


@doc raw"""
    Normal(B::Beta{Float64})

Return `Normal` approximation to a `Beta`. 

If ``B\sim\text{Beta}(a,b)`` then 
```math
B\approx\text{Normal}\left(\frac{a}{a+b}, \frac{ab}{(a+b)^2(a+b+1)}\right)
```
The approximation improves as ``a\approx b`` and for large values of ``a`` and ``b``.

An improvement can be obtained by transforming the parameter (not implemented), [Wise's transformation](https://www.jstor.org/stable/2332968).
"""
function Distributions.Normal(B::Beta{Float64})
    a, b = params(B)
    adequate =
        isapprox((a + 1.0) / (a - 1.0), 1.0, atol = 1e-1) &
        isapprox((b + 1.0) / (b - 1.0), 1.0, atol = 1e-1)
    m = a / (a + b)
    v = a * b / ((a + b)^2 * (a + b + 1))
    if !adequate
        @warn "Normal approximation to `Beta($(a), $(b))` may be poor."
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


@doc raw"""
    ℙless(X::Beta{Float64}, Y::Beta{Float64}; δ = 0.0, method = "approx", n::Int = 10_000)

Return `ℙ(X < Y + δ)`.
The argument `method` may be one of:

- `"approx"` (Normal approximation)
- `"montecarlo"` (Monte-Carlo approximation) 
- `"numeric"` (quadrature).

If `method = "montecarlo"` then `n` indicates the number of samples to use. 
"""
function ℙless(
    X::Beta{Float64},
    Y::Beta{Float64},
    δ::Real = 0.0;
    method = "numeric",
    n::Int = 10_000,
    kwargs...,
)
    if method == "approx"
        N1 = Normal(X)
        N2 = Normal(Y)
        return cdf(N1 - N2, δ)
    elseif method == "montecarlo"
        return mean(rand(X, n) .< rand(Y, n))
    elseif method == "numeric"
        f(x) = pdf(X, x) * cdf(Y, x - δ)
        ∫, err = quadgk(f, δ, 1; kwargs...)
        return ∫
    end
end


@doc raw"""
    ℙmax(D::Vector{Beta{Float64}}; n = 10_000)

Probability that each `Beta ∈ D` is maximum via Monte Carlo simulation.
"""
function ℙmax(D::Vector{Beta{Float64}}; n::Int = 10_000)
    r = reduce(hcat, rand.(D, n))
    m = [x[2] for x in findmax(r, dims = 2)[2]]
    c = counts(m, 1:size(r, 2)) ./ size(r, 1)
    return c
end


@doc raw"""
    ℙmin(D::Vector{Beta{Float64}}; n = 10_000)

Probability that each `Beta ∈ D` is minimum via Monte Carlo simulation.
"""
function ℙmin(D::Vector{Beta{Float64}}; n::Int = 10_000)
    r = reduce(hcat, rand.(D, n))
    m = [x[2] for x in findmin(r, dims = 2)[2]]
    c = counts(m, 1:size(r, 2)) ./ size(r, 1)
    return c
end


@doc raw"""
    Pℙless(
        X::BetaBinomial{Float64},
        Y::BetaBinomial{Float64},
        δ::Float64 = 0.0;
        draws::Int = 10_000,
        kwargs...,
    )

Consider the random variable ``I = \mathbb I(X < Y + \delta)``.
Further, define ``P|D = \mathbb E[I|D]``, i.e. ``\mathbb P(X<Y+\delta|D)``.
This function returns a particle approximation to the distribution of ``P|D`` as a function of ``D``.
The result is used in calculation of PPoS, e.g.
```math
\mathbb E_D[\mathbb P(X < Y + \delta|D) > ϵ]
```

Specifically, if ``\theta_j|Y_j=y_j\sim\text{Beta}(a_j+y_j,b_j+n_j-y_j)`` is the posterior for ``\theta_j`` and 
``Z_j|Y_j=y_j\sim\text{BetaBinomial}(m_j,a_j+y_j,b_j+n_j-y_j),\ j=1,2`` are the posterior predictive distributions of 
``Z_j|Y_j=y_j``, then the function returns
```math
P|Z_j,Y_j=y_j \approx M^{-1}\sum_{m=1}^M \delta_{P^{[m]}}(dP)
``` 
where ``P_m = \mathbb P(\theta_1 < \theta_2 + \delta | Z_j = z_j^{[m]}, Y_j = y_j)``.

This is used in the calculation of PPoS as ``\sum_{m=1}^M P_m > \epsilon``.
"""
function Pℙless(
    X::BetaBinomial{Float64},
    Y::BetaBinomial{Float64},
    δ::Float64 = 0.0;
    draws::Int = 10_000,
    kwargs...,
)
    nX, aX, bX = params(X)
    nY, aY, bY = params(Y)
    yX, yY = rand.([X, Y], draws)
    P = zeros(draws)
    for i = 1:draws
        ppX = Beta(aX + yX[i], bX + nX - yX[i])
        ppY = Beta(aY + yY[i], bY + nY - yY[i])
        P[i] = ℙless(ppX, ppY, δ; kwargs...)
    end
    return P
end


@doc raw"""
    function PPoS(
        ϵ::Float64,
        θ₁::BetaBinomial{Float64},
        θ₂::BetaBinomial{Float64},
        δ::Float64 = 0.0;
        kwargs...,
    )

Calcuate ``\mathbb E_{Z|Y}[\mathbb I\{\mathbb P(\theta_1 < \theta_2 + \delta|Z,Y) > \epsilon\}]``, i.e. PPoS.
"""
function PPoS(
    ϵ::Float64,
    θ₁::BetaBinomial{Float64},
    θ₂::BetaBinomial{Float64},
    δ::Float64 = 0.0;
    kwargs...,
)
    P = Pℙless(θ₁, θ₂, δ; kwargs...)
    return mean(P .> ϵ)
end

end # end module
