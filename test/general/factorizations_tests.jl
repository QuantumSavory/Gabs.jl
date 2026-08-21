@testitem "Factorizations" begin
    using Gabs
    using LinearAlgebra

    @testset "williamson" begin
        nmodes = rand(1:5)
        qpairbasis = QuadPairBasis(nmodes)
        qblockbasis = QuadBlockBasis(nmodes)
        
        state_pair = randstate(qpairbasis)
        state_block = randstate(qblockbasis)

        F_pair = williamson(state_pair)
        S_pair, spectrum_pair = F_pair
        F_block = williamson(state_block)
        S_block, spectrum_block = F_block

        @test S_pair == F_pair.S && spectrum_pair == F_pair.spectrum
        @test S_block == F_block.S && spectrum_block == F_block.spectrum

        @test isapprox(S_pair * state_pair.covar * S_pair', Diagonal(repeat(spectrum_pair, inner = 2)), atol = 1e-5)
        @test isapprox(S_block * state_block.covar * S_block', Diagonal(repeat(spectrum_block, 2)), atol = 1e-5)

        @test issymplectic(qpairbasis, S_pair, atol = 1e-5)
        @test issymplectic(qblockbasis, S_block, atol = 1e-5)
    end

    @testset "polar" begin
        nmodes = rand(1:5)
        qpairbasis = QuadPairBasis(nmodes)
        qblockbasis = QuadBlockBasis(nmodes)
        
        uni_pair = randunitary(qpairbasis)
        uni_block = randunitary(qblockbasis)

        F_pair = polar(uni_pair)
        O_pair, P_pair = F_pair
        F_block = polar(uni_block)
        O_block, P_block = F_block

        @test O_pair == F_pair.O && P_pair == F_pair.P
        @test O_block == F_block.O && P_block == F_block.P

        @test isapprox(O_pair * P_pair, uni_pair.symplectic, atol = 1e-5)
        @test isapprox(O_block * P_block, uni_block.symplectic, atol = 1e-5)

        @test issymplectic(qpairbasis, O_pair, atol = 1e-5) & issymplectic(qblockbasis, O_block, atol = 1e-5)
        @test issymplectic(qpairbasis, P_pair, atol = 1e-5) & issymplectic(qblockbasis, P_block, atol = 1e-5)
        @test isapprox(inv(O_pair), transpose(O_pair), atol = 1e-5) && isapprox(inv(O_block), transpose(O_block), atol = 1e-5)
        @test isapprox(P_block, transpose(P_block), atol = 1e-5) && all(i > 0 for i in eigvals(P_block))
        @test isapprox(P_pair, transpose(P_pair), atol = 1e-5) && all(i > 0 for i in eigvals(P_pair))
    end

    @testset "Bloch-Messiah/Euler" begin
        nmodes = rand(1:5)
        qpairbasis = QuadPairBasis(nmodes)
        qblockbasis = QuadBlockBasis(nmodes)
        
        uni_pair = randunitary(qpairbasis)
        S_pair = uni_pair.symplectic
        uni_block = randunitary(qblockbasis)
        S_block = uni_block.symplectic

        F_pair = blochmessiah(uni_pair)
        O_pair, vals_pair, Q_pair = F_pair
        F_block = blochmessiah(uni_block)
        O_block, vals_block, Q_block = F_block

        @test F_block.O == O_block && F_block.values == vals_block && F_block.Q == Q_block
        @test F_pair.O == O_pair && F_pair.values == vals_pair && F_pair.Q == Q_pair
        @test issymplectic(qpairbasis, O_pair, atol = 1e-5) && issymplectic(qblockbasis, O_block, atol = 1e-5)
        @test issymplectic(qpairbasis, Q_pair, atol = 1e-5) && issymplectic(qblockbasis, Q_block, atol = 1e-5)
        @test isapprox(inv(O_pair), transpose(O_pair), atol = 1e-5) && isapprox(inv(O_block), transpose(O_block), atol = 1e-5)
        @test isapprox(inv(Q_pair), transpose(Q_pair), atol = 1e-5) && isapprox(inv(Q_block), transpose(Q_block), atol = 1e-5)

        @test all(x -> x > 0.0, vals_block) && all(x -> x > 0.0, vals_pair)
        D_block = Diagonal(vcat(vals_block, vals_block .^ (-1)))
        D_pair = Diagonal(collect(Iterators.flatten(zip(vals_pair, vals_pair .^ (-1)))))
        @test isapprox(O_block * D_block * Q_block, S_block, atol = 1e-5)
        @test isapprox(O_pair * D_pair * Q_pair, S_pair, atol = 1e-5)
    end
end