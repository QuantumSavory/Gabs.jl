@testitem "Symbolic States" begin
    using Gabs
    using Symbolics
    using StaticArrays
    using LinearAlgebra: det

    nmodes = rand(1:5)
    qpairbasis = QuadPairBasis(nmodes)
    qblockbasis = QuadBlockBasis(nmodes)

    @testset "Symbolic EPR states" begin
        @variables r θ
        nmodes = rand(1:5)
        @variables rs[1:nmodes] thetas[1:nmodes]
        qpairbasis = QuadPairBasis(nmodes)
        qblockbasis = QuadBlockBasis(nmodes)
        state = eprstate(2 * qpairbasis, r, θ)
        @test state isa GaussianState
        @test eprstate(SVector{4*nmodes}, SMatrix{4*nmodes,4*nmodes}, 2*qpairbasis, r, θ) isa GaussianState
        @test iszero(simplify(eprstate(2*qblockbasis, r, θ).covar - changebasis(QuadBlockBasis, state).covar))
        @test iszero(simplify(eprstate(2*qblockbasis, r, θ).mean - changebasis(QuadBlockBasis, state).mean))
        state_pair = eprstate(2*qpairbasis, collect(rs), collect(thetas))
        state_block = eprstate(2*qblockbasis, collect(rs), collect(thetas))
        @test iszero(simplify(state_block.covar - changebasis(QuadBlockBasis, state_pair).covar))
        @test iszero(simplify(state_block.mean - changebasis(QuadBlockBasis, state_pair).mean))
        @test all(isequal.(eprstate(2 * qblockbasis, collect(rs), collect(thetas)).covar, changebasis(QuadBlockBasis, state_pair).covar))
        @test all(isequal.(eprstate(2 * qblockbasis, collect(rs), collect(thetas)).mean, changebasis(QuadBlockBasis, state_pair).mean))
    end

    @testset "Symbolic squeezed states" begin
        @variables r theta
        @variables rs[1:nmodes] thetas[1:nmodes]
        state = squeezedstate(qpairbasis, r, theta)
        @test state isa GaussianState
        @test squeezedstate(SVector{2*nmodes}, SMatrix{2*nmodes,2*nmodes}, qpairbasis, r, theta) isa GaussianState
        @test all(isequal.(squeezedstate(qblockbasis, r, theta).covar, changebasis(QuadBlockBasis, state).covar))
        @test all(isequal.(squeezedstate(qblockbasis, r, theta).mean, changebasis(QuadBlockBasis, state).mean))
        rs_vec = collect(rs)
        thetas_vec = collect(thetas)
        state = squeezedstate(qpairbasis, rs_vec, thetas_vec)
        @test state isa GaussianState
        @test all(isequal.(squeezedstate(qblockbasis, rs_vec, thetas_vec).covar, changebasis(QuadBlockBasis, state).covar))
        @test all(isequal.(squeezedstate(qblockbasis, rs_vec, thetas_vec).mean, changebasis(QuadBlockBasis, state).mean))
    end

    @testset "Symbolic coherent states" begin
        @variables α
        @variables alphas[1:nmodes]
        state_pair = coherentstate(qpairbasis, α)
        state_block = coherentstate(qblockbasis, α)
        @test state_pair isa GaussianState && state_block isa GaussianState
        @test coherentstate(SVector{2*nmodes}, SMatrix{2*nmodes,2*nmodes}, qpairbasis, α) isa GaussianState
        @test coherentstate(qblockbasis, α).covar == changebasis(QuadBlockBasis, state_pair).covar
        @test isequal(coherentstate(qblockbasis, α).mean, changebasis(QuadBlockBasis, state_pair).mean)
        alphas_vec = vcat([real(alphas[i]) for i in 1:nmodes], [imag(alphas[i]) for i in 1:nmodes])
        @test all.(isequal(coherentstate(qblockbasis, alphas_vec).mean, changebasis(QuadBlockBasis, coherentstate(qpairbasis, alphas_vec)).mean))
        @test all.(isequal(coherentstate(qblockbasis, alphas_vec).covar, changebasis(QuadBlockBasis, coherentstate(qpairbasis, alphas_vec)).covar))
    end

    @testset "Symbolic tensor products" begin
        @variables α r θ
        @variables rs[1:nmodes] thetas[1:nmodes]
        state1 = coherentstate(qpairbasis, α)
        state2 = squeezedstate(qpairbasis, r, θ)
        tensor_state = tensor(state1, state2)
        simplified_tensor = simplify(tensor_state.covar)
        simplified_mean = simplify(tensor_state.mean)
        @test iszero(simplified_tensor - tensor_state.covar)
        @test iszero(simplified_mean - tensor_state.mean)
        rs_vec = collect(rs)
        thetas_vec = collect(thetas)
        state_vec = squeezedstate(qpairbasis, rs_vec, thetas_vec)
        tensor_state_arr = tensor(state1, state_vec)
        simplified_tensor_arr = simplify(tensor_state_arr.covar)
        simplified_mean_arr = simplify(tensor_state_arr.mean)
        @test iszero(simplified_tensor_arr - tensor_state_arr.covar)
        @test iszero(simplified_mean_arr - tensor_state_arr.mean)
    end

    @testset "Symbolic thermal states" begin
        @variables n
        state_pair = thermalstate(qpairbasis, n)
        state_block = thermalstate(qblockbasis, n)
        @test state_pair isa GaussianState && state_block isa GaussianState
        @test thermalstate(SVector{2*nmodes}, SMatrix{2*nmodes,2*nmodes}, qpairbasis, n) isa GaussianState
        @test isequal(thermalstate(qblockbasis, n).covar, changebasis(QuadBlockBasis, state_pair).covar)
        @test isequal(thermalstate(qblockbasis, n).mean, changebasis(QuadBlockBasis, state_pair).mean)
        @variables ns[1:nmodes]
        n_vec = [n for _ in 1:nmodes]
        state_pair = thermalstate(qpairbasis, n_vec)
        state_block = thermalstate(qblockbasis, n_vec)
        @test state_pair isa GaussianState && state_block isa GaussianState
        @test thermalstate(SVector{2*nmodes}, SMatrix{2*nmodes, 2*nmodes}, qpairbasis, n_vec) isa GaussianState
        @test all.(isequal(thermalstate(qblockbasis, n_vec).mean, changebasis(QuadBlockBasis, thermalstate(qpairbasis, n_vec)).mean))
        @test all.(isequal(thermalstate(qblockbasis, n_vec).covar, changebasis(QuadBlockBasis, thermalstate(qpairbasis, n_vec)).covar))
    end
end
