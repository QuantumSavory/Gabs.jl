@testitem "Symbolic Unitaries" begin
    using Gabs
    using Symbolics
    using StaticArrays
    using LinearAlgebra: det

    nmodes = rand(1:5)
    qpairbasis = QuadPairBasis(nmodes)
    qblockbasis = QuadBlockBasis(nmodes)

    for (name, op_func, factor) in [("squeeze", squeeze, 1), ("two-mode squeeze", twosqueeze, 2)]
        @testset "Symbolic $name operator" begin
            @variables r theta
            op_pair = op_func(factor * qpairbasis, r, theta)
            op_block = op_func(factor * qblockbasis, r, theta)
            @test op_pair isa GaussianUnitary && op_block isa GaussianUnitary
            @test op_func(Array, factor * qpairbasis, r, theta) isa GaussianUnitary
            @test op_func(SVector{factor * 2 * nmodes}, SMatrix{factor * 2 * nmodes, factor * 2 * nmodes}, factor * qpairbasis, r, theta) isa GaussianUnitary
            @test isequal(op_func(factor * qblockbasis, r, theta).disp, changebasis(QuadBlockBasis, op_pair).disp)
            @test isequal(op_func(factor * qblockbasis, r, theta).symplectic, changebasis(QuadBlockBasis, op_pair).symplectic)
            @variables rs[1:nmodes] thetas[1:nmodes]
            rs_vec = collect(rs)
            thetas_vec = collect(thetas)
            op_pair_arr = op_func(factor * qpairbasis, rs_vec, thetas_vec)
            op_block_arr = op_func(factor * qblockbasis, rs_vec, thetas_vec)
            @test op_pair_arr isa GaussianUnitary && op_block_arr isa GaussianUnitary
            @test op_func(Array, factor * qpairbasis, rs_vec, thetas_vec) isa GaussianUnitary
            @test op_func(SVector{factor * 2 * nmodes}, SMatrix{factor * 2 * nmodes, factor * 2 * nmodes}, factor * qpairbasis, rs_vec, thetas_vec) isa GaussianUnitary
            @test isequal(op_func(factor * qblockbasis, rs_vec, thetas_vec).disp, changebasis(QuadBlockBasis, op_pair_arr).disp)
            @test isequal(op_func(factor * qblockbasis, rs_vec, thetas_vec).symplectic, changebasis(QuadBlockBasis, op_pair_arr).symplectic)
        end
    end

    for (name, op_func, factor) in [("phase-shift", phaseshift, 1), ("beamsplitter", beamsplitter, 2),("displace", displace, 1)]
        @testset "Symbolic $name operator" begin
            @variables theta
            op = op_func(factor * qpairbasis, theta)
            @test op isa GaussianUnitary
            @test op_func(Array, factor * qpairbasis, theta) isa GaussianUnitary
            @test op_func(SVector{factor * 2 * nmodes}, SMatrix{factor * 2 * nmodes, factor * 2 * nmodes}, factor * qpairbasis, theta) isa GaussianUnitary
            @test isequal(op_func(factor * qblockbasis, theta).disp, changebasis(QuadBlockBasis, op).disp)
            @test isequal(op_func(factor * qblockbasis, theta).symplectic, changebasis(QuadBlockBasis, op).symplectic)
            @variables thetas[1:nmodes]
            thetas_vec = collect(thetas)
            op_arr = op_func(factor * qpairbasis, thetas_vec)
            op_block_arr = op_func(factor * qblockbasis, thetas_vec)
            @test op_arr isa GaussianUnitary && op_block_arr isa GaussianUnitary
            @test op_func(Array, factor * qpairbasis, thetas_vec) isa GaussianUnitary
            @test op_func(SVector{factor * 2 * nmodes}, SMatrix{factor * 2 * nmodes, factor * 2 * nmodes}, factor * qpairbasis, thetas_vec) isa GaussianUnitary
            @test isequal(op_func(factor * qblockbasis, thetas_vec).disp, changebasis(QuadBlockBasis, op_arr).disp)
            @test isequal(op_func(factor * qblockbasis, thetas_vec).symplectic, changebasis(QuadBlockBasis, op_arr).symplectic)
        end
    end

    @testset "Symbolic Tensor Products" begin
        @variables alpha1 alpha2
        d1, d2 = displace(qpairbasis, alpha1), displace(qpairbasis, alpha2)
        ds = tensor(d1, d2)
        @test ds isa GaussianUnitary
        @test isequal(ds.disp, (d1 ⊗ d2).disp)
        @test isequal(ds.symplectic, (d1 ⊗ d2).symplectic)
        @test_throws TypeError isapprox(ds, d1 ⊗ d2, atol = 1e-10)  # due to equality checks for Num type in Symbolics.jl
        @test tensor(SVector{4*nmodes}, SMatrix{4*nmodes, 4*nmodes}, d1, d2) isa GaussianUnitary
        @variables theta
        p = phaseshift(qpairbasis, theta)
        @test isequal(tensor(p, tensor(d1, d2)).disp, (p ⊗ d1 ⊗ d2).disp)
        @test isequal(tensor(p, tensor(d1, d2)).symplectic, (p ⊗ d1 ⊗ d2).symplectic)
        p_block = phaseshift(qblockbasis, theta)
        @variables thetas[1:2*nmodes]
        p_blocks = phaseshift(2 * qblockbasis, collect(thetas))
        @test isequal((p_block ⊗ p_block).disp, p_blocks.disp)
        @test isequal(simplify((p_block ⊗ p_block).symplectic), simplify(substitute(p_blocks.symplectic, Dict(thetas[i] => theta for i in eachindex(thetas)))))
    end

    @testset "Symbolic actions" begin
        @variables α1 α2 r theta r1 theta1
        d = displace(qpairbasis, α1)
        v = vacuumstate(qpairbasis)
        s = squeezedstate(qpairbasis, r, theta)
        c = coherentstate(qpairbasis,α1)
        @test isequal((d * v).mean, c.mean)
        @test isequal((d * v).covar, c.covar)
        @test_broken isequal(apply!(v, d).mean, c.mean)
        @test_broken isequal(apply!(v, d).covar, c.covar)
        @test isequal(apply!(s, d).mean, c.mean)
        d1, d2 = displace(qpairbasis, α1), displace(qpairbasis, α2)
        v1, v2 = vacuumstate(qpairbasis), vacuumstate(qpairbasis)
        s1, s2 = squeezedstate(qpairbasis, r, theta), squeezedstate(qpairbasis, r1, theta1)
        c1, c2 = coherentstate(qpairbasis, α1), coherentstate(qpairbasis, α2)
        @test_throws TypeError simplify(d1 * v1) ≈ simplify(c1) # due to equality checks for Num type in Symbolics.jl
        @test isequal(simplify(d1 * v1).mean, simplify(c1).mean)
        @test isequal(simplify(d1 * v1).covar, simplify(c1).covar)
        @test isequal(apply!(s1, d1).mean, simplify(c1).mean)
        @test isequal(apply!(s2, d1).mean, simplify(c1).mean)
    end
end
