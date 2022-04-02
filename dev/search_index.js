var documenterSearchIndex = {"docs":
[{"location":"index.html#BetaModels.jl-Documentation","page":"Home","title":"BetaModels.jl Documentation","text":"","category":"section"},{"location":"index.html#API","page":"Home","title":"API","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"Modules = [BetaModels]\nOrder   = [:constant, :type, :function, :macro]","category":"page"},{"location":"index.html#Distributions.Normal-Tuple{Beta{Float64}}","page":"Home","title":"Distributions.Normal","text":"Normal(B::Beta{Float64})\n\nReturn Normal approximation to a Beta. \n\nIf BsimtextBeta(ab) then \n\nBapproxtextNormalleft(fracaa+b fracab(a+b)^2(a+b+1)right)\n\nThe approximation improves as aapprox b and for large values of a and b.\n\nAn improvement can be obtained by transforming the parameter (not implemented), Wise's transformation.\n\n\n\n\n\n","category":"method"},{"location":"index.html#Base.:--Tuple{Normal, Normal}","page":"Home","title":"Base.:-","text":"Base.:-(X::Normal{<:Real}, Y::Normal{<:Real})\n\nSubtract a normal random variable Y from another normal random variable X according to\n\ntextNormal(mu sigma^2) - textNormal(nu omega^2) sim textNormal(mu-nu sigma^2+omega^2)\n\n\n\n\n\n","category":"method"},{"location":"index.html#Base.:--Tuple{Normal}","page":"Home","title":"Base.:-","text":"Base.:-(X::Normal{<:Real})\n\nReturn the negative of a Normal random variable, i.e. X sim textNormal(musigma^2) then return -X sim textNormal(-musigma^2).\n\n\n\n\n\n","category":"method"},{"location":"index.html#BetaModels.PPoS","page":"Home","title":"BetaModels.PPoS","text":"function PPoS(\n    ϵ::Float64,\n    θ₁::BetaBinomial{Float64},\n    θ₂::BetaBinomial{Float64},\n    δ::Float64 = 0.0;\n    kwargs...,\n)\n\nCalcuate mathbb E_ZYmathbb Imathbb P(theta_1  theta_2 + deltaZY)  epsilon, i.e. PPoS.\n\n\n\n\n\n","category":"function"},{"location":"index.html#BetaModels.Pℙless","page":"Home","title":"BetaModels.Pℙless","text":"Pℙless(\n    X::BetaBinomial{Float64},\n    Y::BetaBinomial{Float64},\n    δ::Float64 = 0.0;\n    draws::Int = 10_000,\n    kwargs...,\n)\n\nConsider the random variable I = mathbb I(X  Y + delta). Further, define PD = mathbb EID, i.e. mathbb P(XY+deltaD). This function returns a particle approximation to the distribution of PD as a function of D. The result is used in calculation of PPoS, e.g.\n\nmathbb E_Dmathbb P(X  Y + deltaD)  ϵ\n\nSpecifically, if theta_jY_j=y_jsimtextBeta(a_j+y_jb_j+n_j-y_j) is the posterior for theta_j and  Z_jY_j=y_jsimtextBetaBinomial(m_ja_j+y_jb_j+n_j-y_j) j=12 are the posterior predictive distributions of  Z_jY_j=y_j, then the function returns\n\nPZ_jY_j=y_j approx M^-1sum_m=1^M delta_P^m(dP)\n\nwhere P_m = mathbb P(theta_1  theta_2 + delta  Z_j = z_j^m Y_j = y_j).\n\nThis is used in the calculation of PPoS as sum_m=1^M P_m  epsilon.\n\n\n\n\n\n","category":"function"},{"location":"index.html#BetaModels.ℙless","page":"Home","title":"BetaModels.ℙless","text":"ℙless(X::Beta{Float64}, Y::Beta{Float64}; δ = 0.0, method = \"approx\", n::Int = 10_000)\n\nReturn ℙ(X < Y + δ). The argument method may be one of:\n\n\"approx\" (Normal approximation)\n\"montecarlo\" (Monte-Carlo approximation) \n\"numeric\" (quadrature).\n\nIf method = \"montecarlo\" then n indicates the number of samples to use. \n\n\n\n\n\n","category":"function"},{"location":"index.html#BetaModels.ℙmax-Tuple{Vector{Beta{Float64}}}","page":"Home","title":"BetaModels.ℙmax","text":"ℙmax(D::Vector{Beta{Float64}}; n = 10_000)\n\nProbability that each Beta ∈ D is maximum via Monte Carlo simulation.\n\n\n\n\n\n","category":"method"},{"location":"index.html#BetaModels.ℙmin-Tuple{Vector{Beta{Float64}}}","page":"Home","title":"BetaModels.ℙmin","text":"ℙmin(D::Vector{Beta{Float64}}; n = 10_000)\n\nProbability that each Beta ∈ D is minimum via Monte Carlo simulation.\n\n\n\n\n\n","category":"method"},{"location":"index.html#Index","page":"Home","title":"Index","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"","category":"page"}]
}
