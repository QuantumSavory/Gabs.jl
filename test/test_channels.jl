@testitem "Channels" begin
    using Gabs
    using StaticArrays

    nmodes = rand(1:5)
    qpairbasis = QuadPairBasis(nmodes)
    qblockbasis = QuadBlockBasis(nmodes)
    noise = rand(2*nmodes, 2*nmodes)
    noise_ds = [noise zeros(2*nmodes,2*nmodes); zeros(2*nmodes,2*nmodes) noise]
    T = zeros(2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(2*nmodes), j in Base.OneTo(2*nmodes)
        if (j == 2*i-1) || (j + 2*nmodes == 2*i)
            T[i,j] = 1
        end
    end
    T_ds = zeros(4*nmodes, 4*nmodes)
    @inbounds for i in Base.OneTo(4*nmodes), j in Base.OneTo(4*nmodes)
        if (j == 2*i-1) || (j + 2*(2*nmodes) == 2*i)
            T_ds[i,j] = 1
        end
    end

    @testset "displacement operator" begin
        alpha = rand(ComplexF64)
        alphas = rand(ComplexF64, nmodes)
        op_pair = displace(qpairbasis, alpha, noise)
        op_block = displace(qblockbasis, alpha, noise)
        @test op_pair isa GaussianChannel && op_block isa GaussianChannel
        @test displace(SVector{2*nmodes}, SMatrix{2*nmodes,2*nmodes}, qpairbasis, alpha, noise) isa GaussianChannel
        @test displace(Array, qpairbasis, alpha, noise) isa GaussianChannel
        @test displace(qblockbasis, alpha, T*noise*transpose(T)) == changebasis(QuadBlockBasis, op_pair)
        @test displace(qblockbasis, alphas, T*noise*transpose(T)) == changebasis(QuadBlockBasis, displace(qpairbasis, alphas, noise))
    end

    @testset "squeeze operator" begin
        r, theta = rand(Float64), rand(Float64)
        rs, thetas = rand(Float64, nmodes), rand(Float64, nmodes)
        op_pair = squeeze(qpairbasis, r, theta, noise)
        op_block = squeeze(qblockbasis, r, theta, noise)
        @test op_pair isa GaussianChannel && op_block isa GaussianChannel
        @test squeeze(SVector{2*nmodes}, SMatrix{2*nmodes,2*nmodes}, qpairbasis, r, theta, noise) isa GaussianChannel
        @test squeeze(Array, qpairbasis, r, theta, noise) isa GaussianChannel
        @test squeeze(qblockbasis, r, theta, T*noise*transpose(T)) == changebasis(QuadBlockBasis, op_pair)
        @test squeeze(qblockbasis, rs, thetas, T*noise*transpose(T)) == changebasis(QuadBlockBasis, squeeze(qpairbasis, rs, thetas, noise))
    end

    @testset "two-mode squeeze operator" begin
        r, theta = rand(Float64), rand(Float64)
        rs, thetas = rand(Float64, nmodes), rand(Float64, nmodes)
        op = twosqueeze(2*qpairbasis, r, theta, noise_ds)
        @test op isa GaussianChannel
        @test twosqueeze(SVector{4*nmodes}, SMatrix{4*nmodes,4*nmodes}, 2*qpairbasis, r, theta, noise_ds) isa GaussianChannel
        @test twosqueeze(Array, 2*qpairbasis, r, theta, noise_ds) isa GaussianChannel
        @test twosqueeze(2*qblockbasis, r, theta, T_ds*noise_ds*transpose(T_ds)) == changebasis(QuadBlockBasis, op)
        @test twosqueeze(2*qblockbasis, rs, thetas, T_ds*noise_ds*transpose(T_ds)) == changebasis(QuadBlockBasis, twosqueeze(2*qpairbasis, rs, thetas, noise_ds))
    end

    @testset "phase-shift operator" begin
        theta = rand(Float64)
        thetas = rand(Float64, nmodes)
        op = phaseshift(qpairbasis, theta, noise)
        @test op isa GaussianChannel
        @test phaseshift(SVector{2*nmodes}, SMatrix{2*nmodes,2*nmodes}, qpairbasis, theta, noise) isa GaussianChannel
        @test phaseshift(Array, qpairbasis, theta, noise) isa GaussianChannel
        @test phaseshift(qblockbasis, theta, T*noise*transpose(T)) == changebasis(QuadBlockBasis, op)
        @test phaseshift(qblockbasis, thetas, T*noise*transpose(T)) == changebasis(QuadBlockBasis, phaseshift(qpairbasis, thetas, noise))
    end

    @testset "beamsplitter operator" begin
        theta = rand(Float64)
        thetas = rand(Float64, nmodes)
        op = beamsplitter(2*qpairbasis, theta, noise_ds)
        @test op isa GaussianChannel
        @test beamsplitter(SVector{4*nmodes}, SMatrix{4*nmodes,4*nmodes}, 2*qpairbasis, theta, noise_ds) isa GaussianChannel
        @test beamsplitter(Array, 2*qpairbasis, theta, noise_ds) isa GaussianChannel
        @test beamsplitter(2*qblockbasis, theta, T_ds*noise_ds*transpose(T_ds)) == changebasis(QuadBlockBasis, op)
        @test beamsplitter(2*qblockbasis, thetas, T_ds*noise_ds*transpose(T_ds)) == changebasis(QuadBlockBasis, beamsplitter(2*qpairbasis, thetas, noise_ds))
    end

    @testset "attenuator channel" begin
        theta = rand(Float64)
        thetas = rand(Float64, nmodes)
        n = rand(1:10)
        ns = rand(1:10, nmodes)
        op_pair = attenuator(qpairbasis, theta, n)
        op_block = attenuator(qblockbasis, theta, n)
        @test op_pair isa GaussianChannel && op_block isa GaussianChannel
        @test attenuator(SVector{2*nmodes}, SMatrix{2*nmodes,2*nmodes}, qpairbasis, theta, n) isa GaussianChannel
        @test attenuator(Array, qpairbasis, theta, n) isa GaussianChannel
        @test op_pair == changebasis(QuadPairBasis, op_block) && op_block == changebasis(QuadBlockBasis, op_pair)
        @test op_pair == changebasis(QuadPairBasis, op_pair) && op_block == changebasis(QuadBlockBasis, op_block)
        @test attenuator(qblockbasis, theta, n) == changebasis(QuadBlockBasis, op_pair)
        @test attenuator(qblockbasis, thetas, ns) == changebasis(QuadBlockBasis, attenuator(qpairbasis, thetas, ns))
        @test isgaussian(op_pair, atol = 1e-4)
    end

    @testset "amplifier channel" begin
        r = rand(Float64)
        rs = rand(Float64, nmodes)
        n = rand(1:10)
        ns = rand(1:10, nmodes)
        op_pair = amplifier(qpairbasis, r, n)
        op_block = amplifier(qblockbasis, r, n)
        @test op_pair isa GaussianChannel && op_block isa GaussianChannel
        @test amplifier(SVector{2*nmodes}, SMatrix{2*nmodes,2*nmodes}, qpairbasis, r, n) isa GaussianChannel
        @test amplifier(Array, qpairbasis, r, n) isa GaussianChannel
        @test op_pair == changebasis(QuadPairBasis, op_block) && op_block == changebasis(QuadBlockBasis, op_pair)
        @test op_pair == changebasis(QuadPairBasis, op_pair) && op_block == changebasis(QuadBlockBasis, op_block)
        @test amplifier(qblockbasis, r, n) == changebasis(QuadBlockBasis, op_pair)
        @test amplifier(qblockbasis, rs, ns) == changebasis(QuadBlockBasis, amplifier(qpairbasis, rs, ns))
        @test isgaussian(op_pair, atol = 1e-4)
    end
    
    @testset "tensor products" begin
        alpha1, alpha2 = rand(ComplexF64), rand(ComplexF64)
        d1, d2 = displace(qpairbasis, alpha1, noise), displace(qpairbasis, alpha2, noise)
        ds = tensor(d1, d2)
        @test ds isa GaussianChannel
        @test ds == d1 ⊗ d2
        @test isapprox(ds, d1 ⊗ d2, atol = 1e-10)
        @test tensor(SVector{4*nmodes}, SMatrix{4*nmodes,4*nmodes}, d1, d2) isa GaussianChannel

        r, theta = rand(Float64), rand(Float64)
        p = phaseshift(qpairbasis, theta, noise)
        @test tensor(tensor(p, d1), d2) == p ⊗ d1 ⊗ d2

        p_block = phaseshift(qblockbasis, theta, T * noise * transpose(T))
        p_blocks = phaseshift(2*qblockbasis, repeat([theta], 2*nmodes), T_ds * noise_ds * transpose(T_ds))
        @test p_block ⊗ p_block == p_blocks

        dstatic = displace(SVector{2*nmodes}, SMatrix{2*nmodes,2*nmodes}, qpairbasis, alpha1, noise)
        tpstatic = dstatic ⊗ dstatic ⊗ dstatic
        @test tpstatic.disp isa SVector{6*nmodes}
        @test tpstatic.transform isa SMatrix{6*nmodes,6*nmodes}
        @test tpstatic.noise isa SMatrix{6*nmodes,6*nmodes}
        tp = dstatic ⊗ d1 ⊗ dstatic
        @test tp.disp isa Vector
        @test tp.transform isa Matrix
        @test tp.noise isa Matrix
    end

    @testset "actions" begin
        z = zeros(2*nmodes,2*nmodes)
        alpha = rand(ComplexF64)
        d = displace(qpairbasis, alpha, z)
        v = vacuumstate(qpairbasis)
        c = coherentstate(qpairbasis, alpha)
        @test d * v == c
        @test isapprox(d * v, c, atol = 1e-10)
        @test apply!(v, d) == c

        v1, v2 = vacuumstate(qpairbasis), vacuumstate(qpairbasis)
        alpha1, alpha2 = rand(ComplexF64), rand(ComplexF64)
        d1, d2 = displace(qpairbasis, alpha1, z), displace(qpairbasis, alpha2, z)
        c1, c2 = coherentstate(qpairbasis, alpha1), coherentstate(qpairbasis, alpha2)
        @test (d1 ⊗ d2) * (v1 ⊗ v2) == c1 ⊗ c2
    end
end