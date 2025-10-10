@testitem "States" begin
    using Gabs
    using StaticArrays
    using LinearAlgebra: det

    nmodes = rand(1:5)
    qpairbasis = QuadPairBasis(nmodes)
    qblockbasis = QuadBlockBasis(nmodes)

    @testset "vacuum states" begin
        state, array_state, static_state = vacuumstate(qpairbasis), vacuumstate(Array, qpairbasis), vacuumstate(SVector{2*nmodes}, SMatrix{2*nmodes,2*nmodes}, qpairbasis)
        @test state isa GaussianState && array_state isa GaussianState && static_state isa GaussianState
        @test vacuumstate(qblockbasis) == changebasis(QuadBlockBasis, state)
        @test state.ħ == 2 && array_state.ħ == 2 && static_state.ħ == 2
    end

    @testset "thermal states" begin
        n = rand(1:5)
        ns = rand(1:5, nmodes)
        state_pair = thermalstate(qpairbasis, n)
        state_block = thermalstate(qblockbasis, n)
        @test state_pair isa GaussianState && state_block isa GaussianState
        @test thermalstate(SVector{2*nmodes}, SMatrix{2*nmodes,2*nmodes}, qpairbasis, n) isa GaussianState
        @test thermalstate(qblockbasis, n) == changebasis(QuadBlockBasis, state_pair)
        @test state_pair == changebasis(QuadPairBasis, state_block) && state_block == changebasis(QuadBlockBasis, state_pair)
        @test state_pair == changebasis(QuadPairBasis, state_pair) && state_block == changebasis(QuadBlockBasis, state_block)
        @test thermalstate(qblockbasis, ns) == changebasis(QuadBlockBasis, thermalstate(qpairbasis, ns))
        @test isgaussian(state_pair, atol = 1e-4)
        @test state_pair.ħ == 2 && state_block.ħ == 2
    end

    @testset "coherent states" begin
        alpha = rand(ComplexF64)
        alphas = rand(ComplexF64, nmodes)
        state_pair = coherentstate(qpairbasis, alpha)
        state_block = coherentstate(qblockbasis, alpha)
        @test state_pair isa GaussianState && state_block isa GaussianState
        @test coherentstate(SVector{2*nmodes}, SMatrix{2*nmodes,2*nmodes}, qpairbasis, alpha) isa GaussianState
        @test coherentstate(qblockbasis, alpha) == changebasis(QuadBlockBasis, state_pair)
        @test state_pair == changebasis(QuadPairBasis, state_block) && state_block == changebasis(QuadBlockBasis, state_pair)
        @test state_pair == changebasis(QuadPairBasis, state_pair) && state_block == changebasis(QuadBlockBasis, state_block)
        @test coherentstate(qblockbasis, alphas) == changebasis(QuadBlockBasis, coherentstate(qpairbasis, alphas))
        @test isgaussian(state_pair, atol = 1e-4)
        @test state_pair.ħ == 2 && state_block.ħ == 2
    end

    @testset "squeezed states" begin
        r, theta = rand(Float64), rand(Float64)
        rs, thetas = rand(Float64, nmodes), rand(Float64, nmodes)
        state, array_state, static_state = squeezedstate(qpairbasis, r, theta), squeezedstate(Array, qpairbasis, r, theta), squeezedstate(SVector{2*nmodes}, SMatrix{2*nmodes,2*nmodes}, qpairbasis, r, theta)
        @test state isa GaussianState && array_state isa GaussianState && static_state isa GaussianState
        @test squeezedstate(qblockbasis, r, theta) == changebasis(QuadBlockBasis, state)
        @test squeezedstate(qblockbasis, rs, thetas) == changebasis(QuadBlockBasis, squeezedstate(qpairbasis, rs, thetas))
        @test state.ħ == 2 && array_state.ħ == 2 && static_state.ħ == 2
    end

    @testset "epr states" begin
        r, theta = rand(Float64), rand(Float64)
        rs, thetas = rand(Float64, nmodes), rand(Float64, nmodes)
        state, array_state, static_state = eprstate(2*qpairbasis, r, theta), eprstate(Array, 2*qpairbasis, r, theta), eprstate(SVector{4*nmodes}, SMatrix{4*nmodes,4*nmodes}, 2*qpairbasis, r, theta)
        @test state isa GaussianState && array_state isa GaussianState && static_state isa GaussianState
        @test eprstate(SVector{4*nmodes}, SMatrix{4*nmodes,4*nmodes}, 2*qpairbasis, r, theta) isa GaussianState
        @test eprstate(2*qblockbasis, r, theta) == changebasis(QuadBlockBasis, state)
        @test eprstate(2*qblockbasis, rs, thetas) == changebasis(QuadBlockBasis, eprstate(2*qpairbasis, rs, thetas))
        @test state.ħ == 2 && array_state.ħ == 2 && static_state.ħ == 2
    end

    @testset "tensor products" begin
        v = vacuumstate(qpairbasis)
        vs = tensor(v, v)
        @test vs isa GaussianState
        @test tensor(SVector{4*nmodes}, SMatrix{4*nmodes,4*nmodes}, v, v) isa GaussianState
        @test vs == v ⊗ v
        @test isapprox(vs, v ⊗ v, atol = 1e-10)

        alpha = rand(ComplexF64)
        c = coherentstate(qpairbasis, alpha)
        @test tensor(c, tensor(v, v)) == c ⊗ v ⊗ v

        r, theta = rand(Float64), rand(Float64)
        sq = squeezedstate(qblockbasis, r, theta)
        sqs = squeezedstate(2*qblockbasis, repeat([r], 2*nmodes), repeat([theta], 2*nmodes))
        @test sq ⊗ sq == sqs

        vstatic = vacuumstate(SVector{2*nmodes}, SMatrix{2*nmodes,2*nmodes}, qpairbasis)
        tpstatic = vstatic ⊗ vstatic ⊗ vstatic
        @test tpstatic.mean isa SVector{6*nmodes}
        @test tpstatic.covar isa SMatrix{6*nmodes,6*nmodes}
        tp = vstatic ⊗ v ⊗ vstatic
        @test tp.mean isa Vector
        @test tp.covar isa Matrix
    end

    @testset "partial trace" begin
        qpairbasis, qblockbasis = QuadPairBasis(1), QuadBlockBasis(1)
        alpha = rand(Float64)
        r, theta = rand(Float64), rand(Float64)
        n = rand(Int)

        for basis in [qpairbasis, qblockbasis]
            s1, s2, s3 = coherentstate(basis, alpha), squeezedstate(basis, r, theta), thermalstate(basis, n)
            state = s1 ⊗ s2 ⊗ s3
            @test ptrace(state, 1) == s2 ⊗ s3
            @test ptrace(state, 2) == s1 ⊗ s3
            @test ptrace(state, 3) == s1 ⊗ s2
            @test ptrace(state, [1, 2]) == s3
            @test ptrace(state, [1, 3]) == s2
            @test ptrace(state, [2, 3]) == s1
            @test_throws ArgumentError ptrace(state, [1, 2, 3, 4])

            sstatic = coherentstate(SVector{2}, SMatrix{2,2}, basis, alpha)
            tpstatic = sstatic ⊗ sstatic ⊗ sstatic
            @test ptrace(tpstatic, 1) == sstatic ⊗ sstatic
            @test ptrace(tpstatic, [1,3]) == sstatic

            @test ptrace(SVector{2}, SMatrix{2,2}, state, [1, 3]) isa GaussianState
            @test ptrace(SVector{4}, SMatrix{4,4}, state, 1) isa GaussianState

            eprstates = eprstate(basis ⊕ basis ⊕ basis ⊕ basis, r, theta)

            @test ptrace(eprstates, [1, 2]) == eprstate(basis ⊕ basis, r, theta)
        end
    end

    @testset "embed" begin
        qpairbasis = QuadPairBasis(1)
        qblockbasis = QuadBlockBasis(1)
    
        α = rand(Float64) + im * randn(Float64)
        r, θ = rand(Float64), 2π * rand(Float64)
        n = rand(1:5)
    
        for basis in [qpairbasis, qblockbasis]
            s1 = coherentstate(basis, α)
            s2 = squeezedstate(basis, r, θ)
            s3 = thermalstate(basis, n)
            full_basis = basis ⊕ basis ⊕ basis
            state = s1 ⊗ s2 ⊗ s3
    
            @test embed(full_basis, 1, s1) == s1 ⊗ vacuumstate(basis) ⊗ vacuumstate(basis)
            @test embed(full_basis, 2, s2) == vacuumstate(basis) ⊗ s2 ⊗ vacuumstate(basis)
            @test embed(full_basis, 3, s3) == vacuumstate(basis) ⊗ vacuumstate(basis) ⊗ s3
            @test embed(full_basis, [1,2], s1 ⊗ s2) == s1 ⊗ s2 ⊗ vacuumstate(basis)
            @test embed(full_basis, [1,3], s1 ⊗ s3) == s1 ⊗ vacuumstate(basis) ⊗ s3
            @test embed(full_basis, [2,3], s2 ⊗ s3) == vacuumstate(basis) ⊗ s2 ⊗ s3

            @test ptrace(embed(full_basis, 2, s1), 1) == s1
            @test ptrace(embed(full_basis, [2,3], s1 ⊗ s2), [1]) == s1 ⊗ s2
    
            sstatic = coherentstate(SVector{2}, SMatrix{2,2}, basis, α)
            e_static = embed(full_basis, 2, sstatic)
            @test e_static isa GaussianState
            @test ptrace(e_static, 1) == sstatic
    
            @test_throws AssertionError embed(full_basis, [1,2,3,4], s1 ⊗ s2 ⊗ s3)
            @test_throws AssertionError embed(full_basis, [1, 2], s1)  # wrong number of modes
        end
    end    

    @testset "symplectic spectrum" begin
        nmodes = rand(1:5)
        qpairbasis = QuadPairBasis(nmodes)
        qblockbasis = QuadBlockBasis(nmodes)

        s_qpair = randstate(qpairbasis)
        s_qblock = randstate(qblockbasis)

        spec_qpair = sympspectrum(s_qpair)
        spec_qblock = sympspectrum(s_qblock)

        @test all(i > 1 || isapprox(i, 1, atol=1e-5) for i in spec_qpair)
        @test all(i > 1 || isapprox(i, 1, atol=1e-5) for i in spec_qblock)

        @test isapprox(det(s_qpair.covar), prod(abs2, spec_qpair), atol=1e-3)
        @test isapprox(det(s_qblock.covar), prod(abs2, spec_qblock), atol=1e-3)
    end
end