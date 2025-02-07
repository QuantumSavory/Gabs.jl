@testitem "Symbolic Unitaries" begin
    using Gabs
    using Symbolics
    using StaticArrays
    using LinearAlgebra: det

    nmodes = rand(1:5)
    qpairbasis = QuadPairBasis(nmodes)
    qblockbasis = QuadBlockBasis(nmodes)

    @testset "Symbolic displacement operator" begin
        @variables α alphas[1:5]
        op_pair = displace(qpairbasis, α)
        op_block = displace(qblockbasis, α)
        @test op_pair isa GaussianUnitary && op_block isa GaussianUnitary
        @test displace(Array, qpairbasis, α) isa GaussianUnitary
        @test displace(SVector{2*nmodes}, SMatrix{2*nmodes,2*nmodes}, qpairbasis, α) isa GaussianUnitary
        @test all.(isequal(op_pair.symplectic, changebasis(QuadBlockBasis, op_pair).symplectic))
        @test all.(isequal(op_pair.symplectic, changebasis(QuadBlockBasis, op_block).symplectic))
        @test all.(isequal(displace(qblockbasis, α).disp, changebasis(QuadBlockBasis, op_pair).disp))
        @test all.(isequal(displace(qblockbasis, α).symplectic, changebasis(QuadBlockBasis, op_pair).symplectic))
        @test issymplectic(qpairbasis, op_pair.symplectic, atol = 1e-4)
        @test isgaussian(op_pair, atol = 1e-4)
        alphas_vec = vcat([real(alphas[i]) for i in 1:nmodes], [imag(alphas[i]) for i in 1:nmodes])
        op_pair = displace(qpairbasis, alphas_vec)
        op_block = displace(qblockbasis, alphas_vec)
        @test op_pair isa GaussianUnitary && op_block isa GaussianUnitary
        @test displace(Array, qpairbasis, alphas_vec) isa GaussianUnitary
        @test displace(SVector{2*nmodes}, SMatrix{2*nmodes,2*nmodes}, qpairbasis, alphas_vec) isa GaussianUnitary
        @test all.(isequal(op_pair.symplectic, changebasis(QuadBlockBasis, op_pair).symplectic))
        @test all.(isequal(op_pair.symplectic, changebasis(QuadBlockBasis, op_block).symplectic))
        @test all.(isequal(displace(qblockbasis, alphas_vec).disp, changebasis(QuadBlockBasis, op_pair).disp))
        @test all.(isequal(displace(qblockbasis, alphas_vec).symplectic, changebasis(QuadBlockBasis, op_pair).symplectic))
        @test issymplectic(qpairbasis, op_pair.symplectic, atol = 1e-4)
        @test isgaussian(op_pair, atol = 1e-4)
    end

    @testset "Symbolic squeeze operator" begin
        @variables r theta
        op_pair = squeeze(qpairbasis, r, theta)
        op_block = squeeze(qblockbasis, r, theta)
        @test op_pair isa GaussianUnitary && op_block isa GaussianUnitary
        @test squeeze(Array, qpairbasis, r, theta) isa GaussianUnitary
        @test squeeze(SVector{2*nmodes}, SMatrix{2*nmodes, 2*nmodes}, qpairbasis, r, theta) isa GaussianUnitary
        @test all.(isequal(squeeze(qblockbasis, r, theta).disp, changebasis(QuadBlockBasis, op_pair).disp))
        @test all.(isequal(squeeze(qblockbasis, r, theta).symplectic, changebasis(QuadBlockBasis, op_pair).symplectic))
        @variables rs[1:nmodes] thetas[1:nmodes]
        rs_vec = collect(rs)
        thetas_vec = collect(thetas)
        op_pair_arr = squeeze(qpairbasis, rs_vec, thetas_vec)
        op_block_arr = squeeze(qblockbasis, rs_vec, thetas_vec)
        @test op_pair_arr isa GaussianUnitary && op_block_arr isa GaussianUnitary
        @test squeeze(Array, qpairbasis, rs_vec, thetas_vec) isa GaussianUnitary
        @test squeeze(SVector{2*nmodes}, SMatrix{2*nmodes, 2*nmodes}, qpairbasis, rs_vec, thetas_vec) isa GaussianUnitary
        @test all.(isequal(squeeze(qblockbasis, rs_vec, thetas_vec).disp, changebasis(QuadBlockBasis, op_pair_arr).disp))
        @test all.(isequal(squeeze(qblockbasis, rs_vec, thetas_vec).symplectic, changebasis(QuadBlockBasis, op_pair_arr).symplectic))
    end

    @testset "Symbolic two-mode squeeze operator" begin
        @variables r theta
        op = twosqueeze(2 * qpairbasis, r, theta)
        @test op isa GaussianUnitary
        @test twosqueeze(Array, 2 * qpairbasis, r, theta) isa GaussianUnitary
        @test twosqueeze(SVector{4*nmodes}, SMatrix{4*nmodes, 4*nmodes}, 2 * qpairbasis, r, theta) isa GaussianUnitary
        @test all.(isequal(twosqueeze(2 * qblockbasis, r, theta).disp, changebasis(QuadBlockBasis, op).disp))
        @test all.(isequal(twosqueeze(2 * qblockbasis, r, theta).symplectic, changebasis(QuadBlockBasis, op).symplectic))
        @variables rs[1:nmodes] thetas[1:nmodes]
        rs_vec = collect(rs)
        thetas_vec = collect(thetas)
        op_arr = twosqueeze(2 * qpairbasis, rs_vec, thetas_vec)
        op_block_arr = twosqueeze(2 * qblockbasis, rs_vec, thetas_vec)
        @test op_arr isa GaussianUnitary && op_block_arr isa GaussianUnitary
        @test twosqueeze(Array, 2 * qpairbasis, rs_vec, thetas_vec) isa GaussianUnitary
        @test twosqueeze(SVector{4*nmodes}, SMatrix{4*nmodes, 4*nmodes}, 2 * qpairbasis, rs_vec, thetas_vec) isa GaussianUnitary
        @test all.(isequal(twosqueeze(2 * qblockbasis, rs_vec, thetas_vec).disp, changebasis(QuadBlockBasis, op_arr).disp))
        @test all.(isequal(twosqueeze(2 * qblockbasis, rs_vec, thetas_vec).symplectic, changebasis(QuadBlockBasis, op_arr).symplectic))
    end

    @testset "Symbolic phase-shift operator" begin
        @variables theta
        op = phaseshift(qpairbasis, theta)
        @test op isa GaussianUnitary
        @test phaseshift(Array, qpairbasis, theta) isa GaussianUnitary
        @test phaseshift(SVector{2*nmodes}, SMatrix{2*nmodes,2*nmodes}, qpairbasis, theta) isa GaussianUnitary
        @test all.(isequal(phaseshift(qblockbasis, theta).disp, changebasis(QuadBlockBasis, op).disp))
        @test all.(isequal(phaseshift(qblockbasis, theta).symplectic, changebasis(QuadBlockBasis, op).symplectic))
        @variables thetas[1:nmodes]
        thetas_vec = collect(thetas)
        op_arr = phaseshift(qpairbasis, thetas_vec)
        op_block_arr = phaseshift(qblockbasis, thetas_vec)
        @test op_arr isa GaussianUnitary && op_block_arr isa GaussianUnitary
        @test phaseshift(Array, qpairbasis, thetas_vec) isa GaussianUnitary
        @test phaseshift(SVector{2*nmodes}, SMatrix{2*nmodes,2*nmodes}, qpairbasis, thetas_vec) isa GaussianUnitary
        @test all.(isequal(phaseshift(qblockbasis, thetas_vec).disp, changebasis(QuadBlockBasis, op_arr).disp))
        @test all.(isequal(phaseshift(qblockbasis, thetas_vec).symplectic, changebasis(QuadBlockBasis, op_arr).symplectic))
    end
end
