@testitem "Quadrature pair basis" begin
    using Gabs
    using StaticArrays

    basis1 = QuadPairBasis(1)
    @testset "symplectic form" begin
        Omega = symplecticform(2*basis1)
        Omega_static = symplecticform(SMatrix{4, 4}, 2*basis1)
        test_Omega = [0.0 1.0 0.0 0.0;
                      -1.0 0.0 0.0 0.0;
                      0.0 0.0 0.0 1.0;
                      0.0 0.0 -1.0 0.0]
        @test isequal(Omega, test_Omega)
        @test Omega_static isa SMatrix
    end

    @testset "wigner functions" begin
        vac = vacuumstate(basis1)
        c = coherentstate(basis1, 1.0)
        @test isapprox(wignerchar(vac, [0.0, 0.0]), 1.0 - 0.0im)
        @test wigner(vac, [rand(), rand()]) > 0.0
        @test wigner(c, [rand(), rand()]) > 0.0
    end

    @testset "batched evaluation" begin
        # A matrix of points has to give exactly what evaluating the columns one
        # at a time gives; the batched form only rearranges the arithmetic.
        for basis in (QuadPairBasis, QuadBlockBasis), nmodes in (1, 2, 3)
            b = basis(nmodes)
            state = randstate(b)
            xs = randn(2 * nmodes, 6)
            @test wigner(state, xs) ≈ [wigner(state, x) for x in eachcol(xs)]
            @test wignerchar(state, xs) ≈ [wignerchar(state, x) for x in eachcol(xs)]

            s2 = randstate(b)
            @test cross_wigner(state, s2, xs) ≈
                  [cross_wigner(state, s2, x) for x in eachcol(xs)]
            @test cross_wignerchar(state, s2, xs) ≈
                  [cross_wignerchar(state, s2, x) for x in eachcol(xs)]

            lc = GaussianLinearCombination(b, [0.6, -0.3, 0.5],
                                           [state, s2, randstate(b)])
            @test wigner(lc, xs) ≈ [wigner(lc, x) for x in eachcol(xs)]
            @test wignerchar(lc, xs) ≈ [wignerchar(lc, x) for x in eachcol(xs)]
        end

        b = QuadPairBasis(1)
        state = coherentstate(b, 1.0 + 0.5im)
        @test length(wigner(state, randn(2, 4))) == 4
        @test_throws ArgumentError wigner(state, randn(4, 4))
        @test_throws ArgumentError wignerchar(state, randn(4, 4))
    end
end