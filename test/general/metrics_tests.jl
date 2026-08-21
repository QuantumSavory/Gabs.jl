@testitem "Metrics" begin
    using Gabs

    basis1 = QuadPairBasis(1)
    basis2 = QuadPairBasis(2)
    # helper functions
    thermal(x) = thermalstate(basis1, x)
    coherent(x) = coherentstate(basis1, x)
    squeezed(x, y) = squeezedstate(basis1, x, y)
    EPR(x, y) = eprstate(basis2, x, y)

    atol = 1e-10
    # sample count
    count = 10

    @testset "Von Neumann entropy" begin
        # rand samples [0, 1), set entropy_vn_thermal(0) to 0
        entropy_vn_thermal(x) = x > 0 ? log1p(x) - x * log(x / (x + 1)) : 0
        X = rand(Float64, count)
        Y = rand(Float64, count)
        Z = X .* 10 + Y .* (10 * im)

        @test isapprox(
            entropy_vn.(thermal.(X .* 100)),
            entropy_vn_thermal.(X .* 100);
            atol=atol)
        @test isapprox(
            entropy_vn.(coherent.(Z)),
            fill(0.0, count);
            atol=atol)
        @test isapprox(
            entropy_vn.(squeezed.(X, Y .* pi)),
            fill(0.0, count);
            atol=atol)
        @test isapprox(
            entropy_vn.(EPR.(X, Y .* pi)),
            fill(0.0, count);
            atol=atol)
    end

    @testset "fidelity" begin
        # the expression is numerically unstable and would need a dozen branches
        function fidelity_thermal_thermal(a, b)
            x = BigFloat(a); y = BigFloat(b);
            return 1/(sqrt((x + 1) * (y + 1)) - sqrt(x * y))
        end
        fidelity_thermal_coherent(x, y) = exp(- abs(y)^2 / (2 * (x + 1))) / sqrt(x + 1)
        fidelity_coherent_coherent(x, y) = exp(- abs(x - y)^2 / 2)
        X = rand(Float64, count)
        Y = rand(Float64, count)
        Z = X .* 10 + Y .* (10 * im)
        W = rand(ComplexF64, count) .* 10

        @test isapprox(
            fidelity.(thermal.(X .* 100), thermal.(Y .* 100)),
            fidelity_thermal_thermal.(X .* 100, Y .* 100);
            atol=atol)
        @test isapprox(
            fidelity.(thermal.(X .* 100), coherent.(W)),
            fidelity_thermal_coherent.(X .* 100, W);
            atol=atol)
        @test isapprox(
            fidelity.(coherent.(Z), coherent.(W)),
            fidelity_coherent_coherent.(Z, W);
            atol=atol)
    end

    @testset "logarithmic negativity" begin
        X = rand(Float64, count)
        Y = rand(Float64, count)

        @test isapprox(
            logarithmic_negativity.(EPR.(X, Y .* pi), 2),
            X .* 2;
            atol=atol)
    end
end
