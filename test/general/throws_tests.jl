@testitem "Throws" begin
    using Gabs
    using CairoMakie

    basis1 = QuadPairBasis(1)
    @testset "type throws" begin
        @test_throws DimensionMismatch GaussianState(basis1, [1.0, 2.0, 3.0], [3.0 4.0; 5.0 6.0])
        @test_throws DimensionMismatch GaussianChannel(basis1, [1.0, 2.0, 3.0], [3.0 4.0; 5.0 6.0], [3.0 4.0; 5.0 6.0])
        @test_throws DimensionMismatch GaussianChannel(basis1, [1.0, 2.0], [3.0 4.0; 5.0 6.0], [3.0 4.0 4.0; 5.0 6.0 4.0])
    end

    @testset "action throws" begin
        v = vacuumstate(basis1)
        ts = twosqueeze(2*basis1, rand(), rand())
        @test_throws DimensionMismatch ts * v
        @test_throws DimensionMismatch apply!(v, ts)
    end

    @testset "plot extension throws" begin
        ts = twosqueeze(2*basis1, rand(), rand())
        @test_throws ArgumentError Makie.heatmap(collect(-3.0:0.25:3.0), collect(-3.0:0.25:3.0), ts)
    end

    @testset "hbar throws" begin
        rs1, rs2 = randstate(basis1, ħ = 1), randstate(basis1)
        ru1, ru2 = randunitary(basis1, ħ = 1), randunitary(basis1)
        rc1, rc2 = randchannel(basis1, ħ = 1), randchannel(basis1)
        
        @test_throws ArgumentError ru2 * rs1
        @test_throws ArgumentError rc1 * rs2
        @test_throws ArgumentError apply!(rs1, ru2)
        @test_throws ArgumentError apply!(rs2, rc1)
        @test_throws ArgumentError rs1 ⊗ rs2
        @test_throws ArgumentError ru1 ⊗ ru2
        @test_throws ArgumentError rc1 ⊗ rc2
    end
end