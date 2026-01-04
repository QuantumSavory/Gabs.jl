@testitem "Unitaries" begin
    using Gabs
    using StaticArrays
    
    nmodes = rand(1:5)
    qpairbasis = QuadPairBasis(nmodes)
    qblockbasis = QuadBlockBasis(nmodes)

    @testset "displacement operator" begin
        alpha = rand(ComplexF64)
        alphas = rand(ComplexF64, nmodes)
        op_pair = displace(qpairbasis, alpha)
        op_block = displace(qblockbasis, alpha)
        @test op_pair isa GaussianUnitary && op_block isa GaussianUnitary
        @test displace(Array, qpairbasis, alpha) isa GaussianUnitary
        @test displace(SVector{2*nmodes}, SMatrix{2*nmodes,2*nmodes}, qpairbasis, alpha) isa GaussianUnitary
        @test op_pair == changebasis(QuadPairBasis, op_block) && op_block == changebasis(QuadBlockBasis, op_pair)
        @test op_pair == changebasis(QuadPairBasis, op_pair) && op_block == changebasis(QuadBlockBasis, op_block)
        @test displace(qblockbasis, alpha) == changebasis(QuadBlockBasis, op_pair)
        @test displace(qblockbasis, alphas) == changebasis(QuadBlockBasis, displace(qpairbasis, alphas))
        @test issymplectic(qpairbasis, op_pair.symplectic, atol = 1e-4)
        @test isgaussian(op_pair, atol = 1e-4)
        @test op_pair.ħ == 2 && op_block.ħ == 2
    end

    @testset "squeeze operator" begin
        r, theta = rand(Float64), rand(Float64)
        rs, thetas = rand(Float64, nmodes), rand(Float64, nmodes)
        op_pair = squeeze(qpairbasis, r, theta)
        op_block = squeeze(qblockbasis, r, theta)
        @test op_pair isa GaussianUnitary && op_block isa GaussianUnitary
        @test squeeze(Array, qpairbasis, r, theta) isa GaussianUnitary
        @test squeeze(SVector{2*nmodes}, SMatrix{2*nmodes,2*nmodes}, qpairbasis, r, theta) isa GaussianUnitary
        @test op_pair == changebasis(QuadPairBasis, op_block) && op_block == changebasis(QuadBlockBasis, op_pair)
        @test op_pair == changebasis(QuadPairBasis, op_pair) && op_block == changebasis(QuadBlockBasis, op_block)
        @test squeeze(qblockbasis, r, theta) == changebasis(QuadBlockBasis, op_pair)
        @test squeeze(qblockbasis, rs, thetas) == changebasis(QuadBlockBasis, squeeze(qpairbasis, rs, thetas))
        @test issymplectic(qpairbasis, op_pair.symplectic, atol = 1e-4)
        @test isgaussian(op_pair, atol = 1e-4)
        @test op_pair.ħ == 2 && op_block.ħ == 2
    end

    @testset "two-mode squeeze operator" begin
        r, theta = rand(Float64), rand(Float64)
        rs, thetas = rand(Float64, nmodes), rand(Float64, nmodes)
        op, op_array, op_static = twosqueeze(2*qpairbasis, r, theta), twosqueeze(Array, 2*qpairbasis, r, theta), twosqueeze(SVector{4*nmodes}, SMatrix{4*nmodes,4*nmodes}, 2*qpairbasis, r, theta)
        @test op isa GaussianUnitary && op_array isa GaussianUnitary && op_static isa GaussianUnitary
        @test twosqueeze(2*qblockbasis, r, theta) == changebasis(QuadBlockBasis, op)
        @test twosqueeze(2*qblockbasis, rs, thetas) == changebasis(QuadBlockBasis, twosqueeze(2*qpairbasis, rs, thetas))
        @test op.ħ == 2 && op_array.ħ == 2 && op_static.ħ == 2
    end

    @testset "phase-shift operator" begin
        theta = rand(Float64)
        thetas = rand(Float64, nmodes)
        op, op_array, op_static = phaseshift(qpairbasis, theta), phaseshift(Array, qpairbasis, theta), phaseshift(SVector{2*nmodes}, SMatrix{2*nmodes,2*nmodes}, qpairbasis, theta)
        @test op isa GaussianUnitary && op_array isa GaussianUnitary && op_static isa GaussianUnitary
        @test phaseshift(qblockbasis, theta) == changebasis(QuadBlockBasis, op)
        @test phaseshift(qblockbasis, thetas) == changebasis(QuadBlockBasis, phaseshift(qpairbasis, thetas))
        @test op.ħ == 2 && op_array.ħ == 2 && op_static.ħ == 2
    end

    @testset "beamsplitter operator" begin
        theta = rand(Float64)
        thetas = rand(Float64, nmodes)
        op, op_array, op_static = beamsplitter(2*qpairbasis, theta), beamsplitter(Array, 2*qpairbasis, theta), beamsplitter(SVector{4*nmodes}, SMatrix{4*nmodes,4*nmodes}, 2*qpairbasis, theta)
        @test op isa GaussianUnitary && op_array isa GaussianUnitary && op_static isa GaussianUnitary
        @test beamsplitter(2*qblockbasis, theta) == changebasis(QuadBlockBasis, op)
        @test beamsplitter(2*qblockbasis, thetas) == changebasis(QuadBlockBasis, beamsplitter(2*qpairbasis, thetas))
        @test op.ħ == 2 && op_array.ħ == 2 && op_static.ħ == 2
    end

    @testset "tensor products" begin
        alpha1, alpha2 = rand(ComplexF64), rand(ComplexF64)
        d1, d2 = displace(qpairbasis, alpha1), displace(qpairbasis, alpha2)
        ds = tensor(d1, d2)
        @test ds isa GaussianUnitary
        @test ds == d1 ⊗ d2
        @test isapprox(ds, d1 ⊗ d2, atol = 1e-10)
        @test tensor(SVector{4*nmodes}, SMatrix{4*nmodes,4*nmodes}, d1, d2) isa GaussianUnitary

        r, theta = rand(Float64), rand(Float64)
        p = phaseshift(qpairbasis, theta)
        @test tensor(p, tensor(d1, d2)) == p ⊗ d1 ⊗ d2

        p_block = phaseshift(qblockbasis, theta)
        p_blocks = phaseshift(2*qblockbasis, repeat([theta], 2*nmodes))
        @test p_block ⊗ p_block == p_blocks

        dstatic = displace(SVector{2*nmodes}, SMatrix{2*nmodes,2*nmodes}, qpairbasis, alpha1)
        tpstatic = dstatic ⊗ dstatic ⊗ dstatic
        @test tpstatic.disp isa SVector{6*nmodes}
        @test tpstatic.symplectic isa SMatrix{6*nmodes,6*nmodes}
        tp = dstatic ⊗ d1 ⊗ dstatic
        @test tp.disp isa Vector
        @test tp.symplectic isa Matrix
    end

    @testset "actions" begin
        alpha = rand(ComplexF64)
        d = displace(qpairbasis, alpha)
        v = vacuumstate(qpairbasis)
        c = coherentstate(qpairbasis, alpha)
        @test d * v == c
        @test apply!(v, d) == c
        
        v1, v2 = vacuumstate(qpairbasis), vacuumstate(qpairbasis)
        alpha1, alpha2 = rand(ComplexF64), rand(ComplexF64)
        d1, d2 = displace(qpairbasis, alpha1), displace(qpairbasis, alpha2)
        c1, c2 = coherentstate(qpairbasis, alpha1), coherentstate(qpairbasis, alpha2)
        @test (d1 ⊗ d2) * (v1 ⊗ v2) == c1 ⊗ c2
    end

    @testset "embed" begin
        α = rand(ComplexF64)
        r, θ = rand(Float64), rand(Float64)

        for basis in (QuadPairBasis(1), QuadBlockBasis(1))
            op = displace(basis, α)
            full_basis = basis ⊕ basis ⊕ basis

            s1, s2, s3 = vacuumstate(basis), vacuumstate(basis), vacuumstate(basis)
            input_state = s1 ⊗ s2 ⊗ s3

            embedded = embed(full_basis, 2, op)
            expected = s1 ⊗ (op * s2) ⊗ s3
            @test embedded.basis == full_basis
            @test embedded * input_state == expected
        end

        for basis in (QuadPairBasis(2), QuadBlockBasis(2))
            single_basis = typeof(basis)(1)
            op_two = squeeze(basis, r, θ)
            full_basis = basis ⊕ single_basis

            s1 = coherentstate(single_basis, rand())
            s2 = thermalstate(single_basis, rand(1:5))
            s3 = vacuumstate(single_basis)
            input_state = s1 ⊗ s2 ⊗ s3

            embedded_two = embed(full_basis, [1, 3], op_two)
            transformed_13 = op_two * (s1 ⊗ s3)
            result = embedded_two * input_state

            @test ptrace(result, 2) == transformed_13
            @test ptrace(result, [1, 3]) == s2
        end

        @test_throws AssertionError embed(QuadPairBasis(3), [1, 2, 3, 4], displace(QuadPairBasis(1), α))
    end
end