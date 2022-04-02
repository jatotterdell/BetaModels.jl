using Test, BetaModels

@testset "BetaModels.jl" begin
    @test typeof(Normal(Beta(1, 1))) == Normal{Float64}
    @test Normal(Beta(1, 1)) == Normal(1 / 2, sqrt(1 / 12))
    @test_warn r"may be poor" Normal(Beta(1, 1))
    @test -Normal(1, 1) == Normal(-1, 1)
    @test Normal(1, 1) - Normal(1, 1) == Normal(0, sqrt(2))
    @test ℙless(Beta(100, 100), Beta(100, 100); method = "approx") == 0.5
    @test ℙless(Beta(1, 1), Beta(1, 1); method = "numeric") == 0.5
    @test ℙless(Beta(2, 1), Beta(1, 2); method = "numeric") ≈ 5 / 6
    @test isapprox(ℙless(Beta(1, 1), Beta(1, 1); method = "montecarlo"), 0.5, atol = 1e-1)
    @test isapprox(ℙmax([Beta(1, 1), Beta(1, 1)]), [0.5, 0.5], atol = 1e-1)
    @test isapprox(ℙmax([Beta(2, 1), Beta(1, 2)]), [5 / 6, 1 / 6], atol = 1e-1)
    @test isapprox(ℙmin([Beta(2, 1), Beta(1, 2)]), [1 / 6, 5 / 6], atol = 1e-1)
end
