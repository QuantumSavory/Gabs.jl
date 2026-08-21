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
end