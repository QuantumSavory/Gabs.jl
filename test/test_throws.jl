@testitem "Throws" begin
    using Gabs
    using CairoMakie

    basis1 = QuadPairBasis(1)
    basis2 = QuadPairBasis(2)

    @testset "type throws" begin
        @test_throws DimensionMismatch GaussianState(basis1, [1.0, 2.0, 3.0], [3.0 4.0; 5.0 6.0])
        @test_throws DimensionMismatch GaussianChannel(basis1, [1.0, 2.0, 3.0], [3.0 4.0; 5.0 6.0], [3.0 4.0; 5.0 6.0])
        @test_throws DimensionMismatch GaussianChannel(basis1, [1.0, 2.0], [3.0 4.0; 5.0 6.0], [3.0 4.0 4.0; 5.0 6.0 4.0])

        iden = displace(basis1, 0.0 + 0.0im)
        @test_throws DimensionMismatch StellarState(zeros(ComplexF64, 2, 2), iden)
        @test_throws ArgumentError StellarState(zeros(ComplexF64, 3), iden)
        @test_throws DimensionMismatch fockstate(basis1, [1, 2])
    end

    @testset "conversion throws" begin
        @test_throws ArgumentError GaussianState(fockstate(basis1, 1))
        @test_throws ArgumentError StellarState(thermalstate(basis1, 2))
    end

    @testset "action throws" begin
        v = vacuumstate(basis1)
        ts = twosqueeze(2*basis1, rand(), rand())
        @test_throws DimensionMismatch ts * v
        @test_throws DimensionMismatch apply!(v, ts)

        x = fockstate(basis1, 1)
        ts = twosqueeze(basis2, rand(), rand())
        @test_throws DimensionMismatch ts * x
        @test_throws DimensionMismatch apply!(x, ts)
        @test_throws DimensionMismatch ts * displace(basis1, 0.0 + 0.0im)
        @test_throws ArgumentError apply!(x, [1, 2, 3], randunitary(basis1))
    end

    @testset "partial trace throws" begin
        x = fockstate(basis1, 1) ⊗ fockstate(basis1, 2)
        @test_throws ArgumentError ptrace(x, 1)
        @test_throws ArgumentError ptrace(x, [1])
    end

    @testset "photon subtraction throws" begin
        @test_throws ArgumentError subtractphoton(fockstate(basis1, 0))
        @test_throws ArgumentError addphoton(fockstate(basis2, 1), 3)
        @test_throws ArgumentError subtractphoton(fockstate(basis2, 1), 0)
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

        x1, x2 = fockstate(basis1, 1, ħ = 1), fockstate(basis1, 1)
        ru1, ru2 = randunitary(basis1, ħ = 1), randunitary(basis1)
        @test_throws ArgumentError ru2 * x1
        @test_throws ArgumentError apply!(x1, ru2)
        @test_throws ArgumentError apply!(x2, ru1)
        @test_throws ArgumentError ru1 * ru2
        @test_throws ArgumentError x1 ⊗ x2
    end
end