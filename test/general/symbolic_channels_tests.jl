@testitem "Symbolic Channels" begin
    using Gabs
    using Symbolics
    using StaticArrays

    nmodes = 2*rand(1:5)
    qpairbasis = QuadPairBasis(nmodes)
    qblockbasis = QuadBlockBasis(nmodes)
    noise = zeros(Num, 2*nmodes, 2*nmodes)

    @variables a b
    @variables as[1:nmodes] bs[1:nmodes]
    as, bs = collect(as), collect(bs)

    test_configs = [
    (displace, (a, noise), (as, noise), "symbolic displacement operator"), 
    (squeeze, (a, b, noise), (as, bs, noise), "symbolic squeeze operator"), 
    (phaseshift, (a, noise), (as, noise), "symbolic phase-shift operator"),
    (twosqueeze, (a, b, noise), (as, bs, noise), "symbolic two-mode squeeze operator"),
    (beamsplitter, (a, noise), (as, noise), "symbolic beamsplitter operator"),
    (amplifier, (a, b), (as, bs), "symbolic amplifier channel"),
    (attenuator, (a, b), (as, bs), "symbolic attenuator channel")
    ]

    for (op, params, multi_params, desc) in test_configs
        @testset "$desc" begin
            op_pair = op(qpairbasis, params...)
            op_block = op(qblockbasis, params...)  
            @test op_pair isa GaussianChannel && op_block isa GaussianChannel
            @test op(SVector{2*nmodes}, SMatrix{2*nmodes,2*nmodes}, qpairbasis, params...) isa GaussianChannel
            @test op(Array, qpairbasis, params...) isa GaussianChannel
            @test_throws TypeError isapprox(op_block, changebasis(QuadBlockBasis, op_pair)) # due to equality checks for Num type in Symbolics.jl

            op_pair_multi = op(qpairbasis, multi_params...)
            op_block_multi = op(qblockbasis, multi_params...)
            @test op_pair_multi isa GaussianChannel && op_block_multi isa GaussianChannel
            @test op(SVector{2*nmodes}, SMatrix{2*nmodes,2*nmodes}, qpairbasis, multi_params...) isa GaussianChannel
            @test op(Array, qpairbasis, multi_params...) isa GaussianChannel
            @test_throws TypeError isapprox(op_block_multi, changebasis(QuadBlockBasis, op_pair_multi)) # due to equality checks for Num type in Symbolics.jl
        end
    end

    @testset "Symbolic tensor products" begin
        noise_ds = [noise zeros(Num, 2*nmodes, 2*nmodes); zeros(Num, 2*nmodes, 2*nmodes) noise]

        d1_pair = displace(qpairbasis, a, noise)
        d2_pair = displace(qpairbasis, b, noise)
        ds_pair = tensor(d1_pair, d2_pair)
        @test ds_pair isa GaussianChannel
        @test_throws TypeError isapprox(ds_pair, d1_pair ⊗ d2_pair) # due to equality checks for Num type in Symbolics.jl
        @test tensor(SVector{4*nmodes}, SMatrix{4*nmodes,4*nmodes}, d1_pair, d2_pair) isa GaussianChannel
        p_pair = phaseshift(qpairbasis, a, noise)
        @test_throws TypeError isapprox(tensor(tensor(p_pair, d1_pair), d2_pair), p_pair ⊗ d1_pair ⊗ d2_pair) # due to equality checks for Num type in Symbolics.jl
        
        d1_block = displace(qblockbasis, a, noise)
        d2_block = displace(qblockbasis, b, noise)
        ds_block = tensor(d1_block, d2_block)
        @test ds_block isa GaussianChannel
        @test_throws TypeError isapprox(ds_block, d1_block ⊗ d2_block) # due to equality checks for Num type in Symbolics.jl
        @test tensor(SVector{4*nmodes}, SMatrix{4*nmodes,4*nmodes}, d1_block, d2_block) isa GaussianChannel
        p_block = phaseshift(qblockbasis, a, noise)
        @test_throws TypeError isapprox(tensor(tensor(p_block, d1_block), d2_block), p_block ⊗ d1_block ⊗ d2_block) # due to equality checks for Num type in Symbolics.jl
    end

    @testset "Symbolic actions" begin
        d = displace(qpairbasis, a, noise)
        v = vacuumstate(qpairbasis)
        c = coherentstate(qpairbasis, b)
        @test_broken isapprox(d * v, c) # due to equality checks for Num type in Symbolics.jl
        @test_broken isapprox(apply!(v, d), c)
        v1, v2 = vacuumstate(qpairbasis), vacuumstate(qpairbasis)
        d1 = displace(qpairbasis, a, noise)
        d2 = displace(qpairbasis, b, noise)
        c1 = coherentstate(qpairbasis, a)
        c2 = coherentstate(qpairbasis, b)
        @test_throws TypeError isapprox((d1 ⊗ d2) * (v1 ⊗ v2), c1 ⊗ c2) # due to equality checks for Num type in Symbolics.jl
    end
end
