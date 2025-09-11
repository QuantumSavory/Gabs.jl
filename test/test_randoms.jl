@testitem "Random objects" begin
    using Gabs
    using StaticArrays
    using LinearAlgebra: eigvals, adjoint

    @testset "random utils" begin
        nmodes = rand(1:5)
        qpairbasis = QuadPairBasis(nmodes)
        qblockbasis = QuadBlockBasis(nmodes)
        U_qpair = Gabs._rand_unitary(qpairbasis)
        U_qblock = Gabs._rand_unitary(qblockbasis)
        @test isapprox(adjoint(U_qpair), inv(U_qpair), atol = 1e-5)
        @test isapprox(adjoint(U_qblock), inv(U_qblock), atol = 1e-5)

        O_qpair = Gabs._rand_orthogonal_symplectic(qpairbasis)
        O_qblock = Gabs._rand_orthogonal_symplectic(qblockbasis)
        @test isapprox(O_qpair', inv(O_qpair), atol = 1e-5)
        @test isapprox(O_qblock', inv(O_qblock), atol = 1e-5)
        @test issymplectic(qpairbasis, O_qpair, atol = 1e-5)
        @test issymplectic(qblockbasis, O_qblock, atol = 1e-5)

        Spassive_qpair = randsymplectic(qpairbasis, passive = true)
        Spassive_qblock = randsymplectic(qblockbasis, passive = true)
        @test isapprox(Spassive_qpair', inv(Spassive_qpair), atol = 1e-5)
        @test issymplectic(qpairbasis, Spassive_qpair, atol = 1e-5)
        @test isapprox(Spassive_qblock', inv(Spassive_qblock), atol = 1e-5)
        @test issymplectic(qblockbasis, Spassive_qblock, atol = 1e-5)

        S_qpair = randsymplectic(qpairbasis)
        S_qblock = randsymplectic(qblockbasis)
        @test issymplectic(qpairbasis, S_qpair, atol = 1e-5)
        @test issymplectic(qblockbasis, S_qblock, atol = 1e-5)

        S_array = randsymplectic(Array, qpairbasis)
        @test issymplectic(qpairbasis, S_array, atol = 1e-5)

        S_sarray = randsymplectic(SArray, qpairbasis)
        @test issymplectic(qpairbasis, S_sarray, atol = 1e-5)

        S_smatrix = randsymplectic(SMatrix, qpairbasis)
        @test issymplectic(qpairbasis, S_smatrix, atol = 1e-5)

        S_sized = randsymplectic(SMatrix{2nmodes,2nmodes}, qpairbasis)
        @test issymplectic(qpairbasis, S_sized, atol = 1e-5)

        S_qpair = randsymplectic(SMatrix{2nmodes,2nmodes}, qpairbasis)
        @test issymplectic(qpairbasis, S_qpair, atol=1e-5)

        S_qblock = randsymplectic(SMatrix{2nmodes,2nmodes}, qblockbasis)
        @test issymplectic(qblockbasis, S_qblock, atol=1e-5)

        S_passive_qpair = randsymplectic(SMatrix{2nmodes,2nmodes}, qpairbasis, passive=true)
        @test isapprox(S_passive_qpair', inv(S_passive_qpair), atol=1e-5)
        @test issymplectic(qpairbasis, S_passive_qpair, atol=1e-5)

        S_passive_qblock = randsymplectic(SMatrix{2nmodes,2nmodes}, qblockbasis, passive=true)
        @test isapprox(S_passive_qblock', inv(S_passive_qblock), atol=1e-5)
        @test issymplectic(qblockbasis, S_passive_qblock, atol=1e-5)
    end

    @testset "random states" begin
        nmodes, ħ = rand(1:5), rand(1:5)
        qpairbasis = QuadPairBasis(nmodes)
        qblockbasis = QuadBlockBasis(nmodes)
        rs_pair = randstate(qpairbasis)
        rs_block = randstate(qblockbasis)
        rc = randchannel(qpairbasis)
        @test rc isa GaussianChannel
        @test rc * rs_pair isa GaussianState
        @test rs_pair.ħ == 2 && rs_block.ħ == 2
        @test isgaussian(rs_pair, atol = 1e-5)
        @test isgaussian(rs_block, atol = 1e-5)

        rspure_pair = randstate(qpairbasis, pure = true, ħ = ħ)
        rspure_block = randstate(qblockbasis, pure = true, ħ = ħ)
        @test isgaussian(rspure_pair, atol = 1e-5)
        @test isapprox(purity(rspure_pair), 1.0, atol = 1e-5)
        @test isgaussian(rspure_block, atol = 1e-5)
        @test isapprox(purity(rspure_block), 1.0, atol = 1e-5)

        rs_array = randstate(Array, qpairbasis)
        rc_array = randchannel(Array, qpairbasis)
        @test rc_array isa GaussianChannel
        @test rc_array.ħ == 2
        @test rc_array * rs_array isa GaussianState
        @test isgaussian(rs_array, atol = 1e-5)

        rs_array = randstate(SArray, qpairbasis)
        rc_array = randchannel(SArray, qpairbasis)
        @test rc_array isa GaussianChannel
        @test rc_array.ħ == 2
        @test rc_array * rs_array isa GaussianState
        @test isgaussian(rs_array, atol = 1e-5)

        rspure_array = randstate(Array, qpairbasis, pure = true)
        @test isgaussian(rspure_array, atol = 1e-5)
        @test isapprox(purity(rspure_array), 1.0, atol = 1e-3)

        rspure_array = randstate(SArray, qpairbasis, pure = true)
        @test isgaussian(rspure_array, atol = 1e-5)
        @test isapprox(purity(rspure_array), 1.0, atol = 1e-3)

        rs_static = randstate(SVector{2*nmodes}, SMatrix{2*nmodes,2*nmodes}, qpairbasis)
        rc_static = randchannel(SVector{2*nmodes}, SMatrix{2*nmodes,2*nmodes}, qpairbasis)
        @test rc_static isa GaussianChannel
        @test rs_static isa GaussianState
        @test rc_static * rs_static isa GaussianState
        @test isgaussian(rs_static, atol = 1e-5)

        rspure_static = randstate(SVector{2*nmodes}, SMatrix{2*nmodes,2*nmodes}, qpairbasis, pure = true)
        @test isgaussian(rspure_static, atol = 1e-5)
        @test isapprox(purity(rspure_static), 1.0, atol = 1e-5)

        rs_static = randstate(SVector, SMatrix, qpairbasis)
        rc_static = randchannel(SVector, SMatrix, qpairbasis)
        @test rc_static isa GaussianChannel
        @test rs_static isa GaussianState
        @test rc_static * rs_static isa GaussianState
        @test isgaussian(rs_static, atol = 1e-5)

        rspure_static = randstate(SVector, SMatrix, qpairbasis, pure = true)
        @test isgaussian(rspure_static, atol = 1e-5)
        @test isapprox(purity(rspure_static), 1.0, atol = 1e-5)

        ρ_sarray = randstate(SArray, qpairbasis)
        @test ρ_sarray isa GaussianState
        @test isgaussian(ρ_sarray, atol=1e-5)
        @test ρ_sarray.ħ == 2

        ρ_svec = randstate(SVector, SMatrix, qpairbasis)
        @test ρ_svec isa GaussianState
        @test isgaussian(ρ_svec, atol=1e-5)

        ρ_sized = randstate(SVector{2nmodes}, SMatrix{2nmodes,2nmodes}, qpairbasis)
        @test ρ_sized isa GaussianState

        @test isgaussian(ρ_sized, atol=1e-5)
        ρ_pure = randstate(SArray, qpairbasis, pure=true)
        @test isapprox(purity(ρ_pure), 1.0, atol=1e-5)
    end

    @testset "random unitaries" begin
        nmodes, ħ = rand(1:5), rand(1:5)
        qpairbasis = QuadPairBasis(nmodes)
        qblockbasis = QuadBlockBasis(nmodes)
        ru = randunitary(qpairbasis, ħ = ħ)
        @test isgaussian(ru, atol = 1e-5)

        rupassive = randunitary(qpairbasis, passive = true)
        @test rupassive.ħ == 2
        @test isapprox(rupassive.symplectic', inv(rupassive.symplectic), atol = 1e-5)
        @test isgaussian(rupassive, atol = 1e-5)

        ru_array = randunitary(Array, qpairbasis)
        @test ru_array.ħ == 2
        @test isgaussian(ru_array, atol = 1e-5)

        ru_array = randunitary(SArray, qpairbasis)
        @test ru_array.ħ == 2
        @test isgaussian(ru_array, atol = 1e-5)

        rupassive_array = randunitary(qpairbasis, passive = true)
        @test isapprox(rupassive_array.symplectic', inv(rupassive_array.symplectic), atol = 1e-5)
        @test isgaussian(rupassive_array, atol = 1e-5)

        ru_static = randunitary(SVector{2*nmodes}, SMatrix{2*nmodes,2*nmodes}, qpairbasis)
        @test ru_static.ħ == 2
        @test isgaussian(ru_static, atol = 1e-5)

        rupassive_static = randunitary(SVector{2*nmodes}, SMatrix{2*nmodes,2*nmodes}, qpairbasis, passive = true)
        @test isapprox(rupassive_static.symplectic', inv(rupassive_static.symplectic), atol = 1e-5)
        @test isgaussian(rupassive_static, atol = 1e-5)

        ru_static = randunitary(SVector, SMatrix, qpairbasis)
        @test ru_static.ħ == 2
        @test isgaussian(ru_static, atol = 1e-5)

        rupassive_static = randunitary(SVector, SMatrix, qpairbasis, passive = true)
        @test isapprox(rupassive_static.symplectic', inv(rupassive_static.symplectic), atol = 1e-5)
        @test isgaussian(rupassive_static, atol = 1e-5)
    end

    @testset "random channels" begin
        nmodes, ħ = rand(1:5), rand(1:5)
        qpairbasis = QuadPairBasis(nmodes)
        qblockbasis = QuadBlockBasis(nmodes)
        rc = randchannel(qpairbasis, ħ = ħ)
        @test isgaussian(rc, atol = 1e-5)

        rc_array = randchannel(Array, qpairbasis)
        @test rc_array.ħ == 2
        @test isgaussian(rc_array, atol = 1e-5)

        rc_static = randchannel(SVector{2*nmodes}, SMatrix{2*nmodes, 2*nmodes}, qpairbasis)
        @test rc_static.ħ == 2
        @test isgaussian(rc_static, atol = 1e-5)

        rc_array = randchannel(SArray, qpairbasis)
        @test rc_array.ħ == 2
        @test isgaussian(rc_array, atol = 1e-5)

        rc_static = randchannel(SVector, SMatrix, qpairbasis)
        @test rc_static.ħ == 2
        @test isgaussian(rc_static, atol = 1e-5)
    end
end
