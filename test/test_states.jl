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
        @test thermalstate(SVector, SMatrix, qpairbasis, n) isa GaussianState
        @test thermalstate(SArray, qpairbasis, n) isa GaussianState
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
        @test coherentstate(SVector, SMatrix, qpairbasis, alpha) isa GaussianState
        @test coherentstate(SArray, qpairbasis, alpha) isa GaussianState
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
        state, array_state, static_state, static_array_state, static_state1 = eprstate(2*qpairbasis, r, theta), eprstate(Array, 2*qpairbasis, r, theta), eprstate(SVector{4*nmodes}, SMatrix{4*nmodes,4*nmodes}, 2*qpairbasis, r, theta), eprstate(SArray, 2*qpairbasis, r, theta), eprstate(SVector, SMatrix, 2*qpairbasis, r, theta)
        @test state isa GaussianState && array_state isa GaussianState && static_state isa GaussianState && static_state1 isa GaussianState && static_array_state isa GaussianState
        @test eprstate(SVector{4*nmodes}, SMatrix{4*nmodes,4*nmodes}, 2*qpairbasis, r, theta) isa GaussianState
        @test eprstate(SVector, SMatrix, 2*qpairbasis, r, theta) isa GaussianState
        @test eprstate(SArray, 2*qpairbasis, r, theta) isa GaussianState
        @test eprstate(2*qblockbasis, r, theta) == changebasis(QuadBlockBasis, state)
        @test eprstate(2*qblockbasis, rs, thetas) == changebasis(QuadBlockBasis, eprstate(2*qpairbasis, rs, thetas))
        @test state.ħ == 2 && array_state.ħ == 2 && static_state.ħ == 2
    end

    @testset "tensor products" begin
        v = vacuumstate(qpairbasis)
        vs = tensor(v, v)
        @test vs isa GaussianState
        @test tensor(SVector{4*nmodes}, SMatrix{4*nmodes,4*nmodes}, v, v) isa GaussianState
        @test tensor(SVector, SMatrix, v, v) isa GaussianState
        @test tensor(SArray, v, v) isa GaussianState
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
        vstatic = vacuumstate(SVector, SMatrix, qpairbasis)
        vstatic = vacuumstate(SArray, qpairbasis)
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

            sstatic = coherentstate(SVector, SMatrix, basis, alpha)
            tpstatic = sstatic ⊗ sstatic ⊗ sstatic
            @test ptrace(tpstatic, 1) == sstatic ⊗ sstatic
            @test ptrace(tpstatic, [1,3]) == sstatic

            sstatic = coherentstate(SArray, basis, alpha)
            tpstatic = sstatic ⊗ sstatic ⊗ sstatic
            @test ptrace(tpstatic, 1) == sstatic ⊗ sstatic
            @test ptrace(tpstatic, [1,3]) == sstatic

            for (T1, T2, subsys) in [
                (SVector{2}, SMatrix{2,2}, [1, 3]),
                (SVector{4}, SMatrix{4,4}, 1),
                (SVector,    SMatrix,      [1, 3]),
                (SVector,    SMatrix,      1),
                (SArray,     nothing,      [1, 3]),
                (SArray,     nothing,      1)
            ]
                if T2 === nothing
                    @test ptrace(T1, state, subsys) isa GaussianState
                else
                    @test ptrace(T1, T2, state, subsys) isa GaussianState
                end
            end

            eprstates = eprstate(basis ⊕ basis ⊕ basis ⊕ basis, r, theta)

            @test ptrace(eprstates, [1, 2]) == eprstate(basis ⊕ basis, r, theta)
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