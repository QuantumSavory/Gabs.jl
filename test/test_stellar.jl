@testitem "Stellar states" begin
    using Gabs
    using LinearAlgebra

    nmodes = rand(1:5)
    qpairbasis = QuadPairBasis(nmodes)
    qblockbasis = QuadBlockBasis(nmodes)

    @testset "fock states" begin
        n = rand(1:3)
        ns = rand(1:3, nmodes)
        state_pair = fockstate(qpairbasis, n)
        state_block = fockstate(qblockbasis, n)
        @test state_pair isa StellarState && state_block isa StellarState
        @test fockstate(Array, qpairbasis, n) isa StellarState
        @test ndims(state_pair.core) == nmodes
        @test size(state_pair.core) == Tuple(fill(n + 1, nmodes))
        @test state_pair.core[CartesianIndex(size(state_pair.core))] == one(ComplexF64)
        @test stellarrank(state_pair) == n * nmodes
        @test stellarrank(fockstate(qpairbasis, ns)) == sum(ns)
        @test iszero(state_pair.gaussian.disp)
        @test state_pair.gaussian.symplectic == Matrix{Float64}(I, 2*nmodes, 2*nmodes)
        @test state_pair.basis == qpairbasis && state_block.basis == qblockbasis
        @test state_pair.ħ == 2 && state_block.ħ == 2
        @test fockstate(qpairbasis, n, ħ = 1).ħ == 1
        @test propertynames(state_pair) == (:basis, :core, :gaussian, :ħ)
    end

    @testset "gaussian unitary group" begin
        op1, op2 = randunitary(qpairbasis), randunitary(qpairbasis)
        prod = op1 * op2
        @test prod isa GaussianUnitary
        @test isapprox(prod.symplectic, op1.symplectic * op2.symplectic, atol = 1e-5)
        @test isapprox(prod.disp, op1.symplectic * op2.disp + op1.disp, atol = 1e-5)
        @test issymplectic(qpairbasis, prod.symplectic, atol = 1e-5)

        v = randstate(qpairbasis)
        @test isapprox((op1 * op2) * v, op1 * (op2 * v), atol = 1e-5)

        for basis in [qpairbasis, qblockbasis]
            op = randunitary(basis)
            iden = op * inv(op)
            @test issymplectic(basis, inv(op).symplectic, atol = 1e-5)
            @test isapprox(iden.symplectic, Matrix{Float64}(I, 2*nmodes, 2*nmodes), atol = 1e-5)
            @test isapprox(iden.disp, zeros(2*nmodes), atol = 1e-5)
            @test isapprox((inv(op) * op).symplectic, Matrix{Float64}(I, 2*nmodes, 2*nmodes), atol = 1e-5)
            @test isapprox((inv(op) * op).disp, zeros(2*nmodes), atol = 1e-5)
            @test inv(op).ħ == op.ħ
        end
    end

    @testset "stellar rank" begin
        n = rand(1:3)
        @test stellarrank(fockstate(qpairbasis, 0)) == 0
        @test stellarrank(randstate(qpairbasis)) == 0
        @test isgaussian(fockstate(qpairbasis, 0), atol = 1e-5)
        @test !isgaussian(fockstate(qpairbasis, n), atol = 1e-5)

        op = randunitary(qpairbasis)
        x = fockstate(qpairbasis, n)
        @test stellarrank(op * x) == stellarrank(x)
    end

    @testset "gaussian conversion" begin
        for basis in [qpairbasis, qblockbasis]
            st = randstate(basis, pure = true)
            x = StellarState(st)
            @test x isa StellarState
            @test stellarrank(x) == 0
            @test isapprox(GaussianState(x), st, rtol = 1e-5)
            @test isapprox(x.gaussian.symplectic * transpose(x.gaussian.symplectic),
                           (2/st.ħ) * st.covar, rtol = 1e-5)
            @test x.ħ == st.ħ

            sq = squeezedstate(basis, 0.5, π/4)
            y = StellarState(sq)
            @test isapprox(y.gaussian.symplectic, squeeze(basis, 0.5, π/4).symplectic, atol = 1e-10)
            @test isapprox(GaussianState(y), sq, atol = 1e-10)
            @test issymplectic(basis, y.gaussian.symplectic, atol = 1e-10)
        end
    end

    @testset "unitary action" begin
        n = rand(1:3)
        for basis in [qpairbasis, qblockbasis]
            x = fockstate(basis, n)
            op = randunitary(basis)

            y = op * x
            @test y isa StellarState
            @test y.core == x.core
            @test isapprox(y.gaussian, op * x.gaussian, atol = 1e-5)

            z = apply!(copy(x), op)
            @test isapprox(z, y, atol = 1e-5)
            @test stellarrank(z) == stellarrank(x)

            single = typeof(basis)(1)
            opsub = randunitary(single)
            idx = rand(1:nmodes)
            @test isapprox(apply!(copy(x), [idx], opsub),
                           embed(basis, [idx], opsub) * x, atol = 1e-5)
            @test isapprox(apply!(copy(x), idx, opsub),
                           apply!(copy(x), [idx], opsub), atol = 1e-5)
        end
    end

    @testset "tensor products" begin
        basis1 = QuadPairBasis(1)
        m, n = rand(1:3), rand(1:3)
        x, y = fockstate(basis1, m), fockstate(basis1, n)

        xy = tensor(x, y)
        @test xy isa StellarState
        @test xy == x ⊗ y
        @test isapprox(xy, fockstate(QuadPairBasis(2), [m, n]), atol = 1e-10)
        @test size(xy.core) == (m + 1, n + 1)
        @test stellarrank(xy) == stellarrank(x) + stellarrank(y)
        @test tensor(Array, Vector{Float64}, Matrix{Float64}, x, y) isa StellarState

        z = fockstate(basis1, 0)
        @test isapprox(x ⊗ y ⊗ z, fockstate(QuadPairBasis(3), [m, n, 0]), atol = 1e-10)
    end

    @testset "embed" begin
        n = rand(1:3)
        for basis in [QuadPairBasis(1), QuadBlockBasis(1)]
            full_basis = basis ⊕ basis ⊕ basis
            x = fockstate(basis, n)
            vac = fockstate(basis, 0)

            @test isapprox(embed(full_basis, 1, x), x ⊗ vac ⊗ vac, atol = 1e-10)
            @test isapprox(embed(full_basis, 2, x), vac ⊗ x ⊗ vac, atol = 1e-10)
            @test isapprox(embed(full_basis, 3, x), vac ⊗ vac ⊗ x, atol = 1e-10)
            @test isapprox(embed(full_basis, [1, 3], x ⊗ x), x ⊗ vac ⊗ x, atol = 1e-10)
            @test stellarrank(embed(full_basis, 2, x)) == stellarrank(x)

            @test_throws AssertionError embed(full_basis, [1, 2], x)
            @test_throws AssertionError embed(full_basis, [1, 2, 3, 4], x ⊗ x ⊗ x ⊗ x)
        end
    end

    @testset "changebasis" begin
        n = rand(1:3)
        x_pair = randunitary(qpairbasis) * fockstate(qpairbasis, n)
        x_block = changebasis(QuadBlockBasis, x_pair)

        @test x_block isa StellarState
        @test x_block.basis isa QuadBlockBasis
        @test x_block.core == x_pair.core
        @test stellarrank(x_block) == stellarrank(x_pair)
        @test isapprox(changebasis(QuadPairBasis, x_block), x_pair, atol = 1e-10)
        @test isapprox(changebasis(QuadBlockBasis, x_block), x_block, atol = 1e-10)
        @test isapprox(changebasis(QuadPairBasis, x_pair), x_pair, atol = 1e-10)
        @test x_block.ħ == x_pair.ħ

        xi = randn(2*nmodes)
        @test isapprox(wigner(x_pair, xi), wigner(x_block, changebasis(QuadBlockBasis, GaussianState(QuadPairBasis(nmodes), xi, Matrix{Float64}(I, 2*nmodes, 2*nmodes))).mean), atol = 1e-8)
    end

    @testset "photon addition and subtraction" begin
        basis1 = QuadPairBasis(1)
        r, theta = rand(Float64), 2π * rand(Float64)
        alpha = rand(ComplexF64)

        vac = fockstate(basis1, 0)
        @test isapprox(addphoton(vac), fockstate(basis1, 1), atol = 1e-10)
        @test isapprox(addphoton(addphoton(vac)), fockstate(basis1, 2), atol = 1e-10)
        @test stellarrank(addphoton(vac)) == 1
        @test isapprox(subtractphoton(addphoton(vac)), vac, atol = 1e-10)
        @test length(subtractphoton(addphoton(vac)).core) == 1

        sq = squeeze(basis1, r, theta) * vac
        @test stellarrank(sq) == 0
        @test stellarrank(addphoton(sq)) == 1
        @test stellarrank(subtractphoton(sq)) == 1
        @test isapprox(addphoton(sq).gaussian, sq.gaussian, atol = 1e-10)
        @test isapprox(sum(abs2, addphoton(sq).core), 1.0, atol = 1e-10)
        @test isapprox(sum(abs2, subtractphoton(sq).core), 1.0, atol = 1e-10)

        coh = displace(basis1, alpha) * vac
        sub = subtractphoton(coh)
        @test stellarrank(sub) == 0
        @test length(sub.core) == 1
        @test isapprox(abs(sub.core[1]), 1.0, atol = 1e-10)
        @test isapprox(sub.gaussian, coh.gaussian, atol = 1e-10)
        @test isapprox(wigner(sub, [0.3, -0.7]), wigner(coh, [0.3, -0.7]), atol = 1e-10)
        @test stellarrank(addphoton(coh)) == 1
    end

    @testset "wigner functions" begin
        r, theta = rand(Float64), 2π * rand(Float64)
        alpha = rand(ComplexF64)
        xi = randn(2*nmodes)

        for basis in [qpairbasis, qblockbasis]
            op = randunitary(basis, passive = true)
            @test isapprox(wigner(op * fockstate(basis, 0), xi),
                           wigner(op * vacuumstate(basis), xi), rtol = 1e-6)

            st = randstate(basis, pure = true)
            @test isapprox(wigner(StellarState(st), st.mean), wigner(st, st.mean), rtol = 1e-6)

            sq = squeezedstate(basis, 0.5, π/4)
            @test isapprox(wigner(StellarState(sq), xi), wigner(sq, xi), rtol = 1e-6)
        end

        basis1 = QuadPairBasis(1)
        @test isapprox(wigner(fockstate(basis1, 0), [0.0, 0.0]), 1/(2π), atol = 1e-10)
        @test isapprox(wigner(fockstate(basis1, 1), [0.0, 0.0]), -1/(2π), atol = 1e-10)
        @test wigner(fockstate(basis1, 1), [0.0, 0.0]) < 0
        @test_throws ArgumentError wigner(fockstate(basis1, 1), zeros(4))
    end

    @testset "stellar function" begin
        basis1 = QuadPairBasis(1)
        z = rand(ComplexF64)
        r = rand(Float64)
        alpha = rand(ComplexF64)

        @test isapprox(stellarfunction(fockstate(basis1, 0), z), 1.0 + 0.0im, atol = 1e-8)
        @test isapprox(stellarfunction(fockstate(basis1, 1), z), z, atol = 1e-8)
        @test isapprox(stellarfunction(fockstate(basis1, 2), z), z^2/sqrt(2), atol = 1e-8)
        @test isapprox(stellarfunction(fockstate(basis1, 1), [z]),
                       stellarfunction(fockstate(basis1, 1), z), atol = 1e-10)

        coh = displace(basis1, alpha) * fockstate(basis1, 0)
        @test isapprox(stellarfunction(coh, z), exp(-abs2(alpha)/2 + alpha*z), atol = 1e-8)

        dis1 = displace(basis1, alpha) * fockstate(basis1, 1)
        @test isapprox(stellarfunction(dis1, z), (z - conj(alpha))*exp(-abs2(alpha)/2 + alpha*z), atol = 1e-8)
        @test isapprox(stellarfunction(dis1, conj(alpha)), 0.0 + 0.0im, atol = 1e-8)

        sq = squeeze(basis1, r, 0.0) * fockstate(basis1, 0)
        @test isapprox(stellarfunction(sq, z), exp(-tanh(r)*z^2/2)/sqrt(cosh(r)), atol = 1e-8)
        @test isapprox(stellarfunction(sq, 0.0 + 0.0im), 1/sqrt(cosh(r)), atol = 1e-8)

        @test isapprox(stellarfunction(addphoton(fockstate(basis1, 0)), z),
                       stellarfunction(fockstate(basis1, 1), z), atol = 1e-8)

        two = fockstate(basis1, 1) ⊗ fockstate(basis1, 1)
        w = rand(ComplexF64)
        @test isapprox(stellarfunction(two, [z, w]), z*w, atol = 1e-8)
        @test_throws DimensionMismatch stellarfunction(fockstate(basis1, 1), [z, w])
    end

    @testset "random stellar states" begin
        rank = rand(0:3)
        for basis in [qpairbasis, qblockbasis]
            x = randstellar(basis, rank)
            @test x isa StellarState
            @test randstellar(Array, basis, rank) isa StellarState
            @test stellarrank(x) == rank
            @test isapprox(sum(abs2, x.core), 1.0, atol = 1e-10)
            @test issymplectic(basis, x.gaussian.symplectic, atol = 1e-5)
            @test randstellar(basis, rank, ħ = 1).ħ == 1
            @test randstellar(basis, rank, passive = true) isa StellarState
            @test x.ħ == 2
        end
        @test isgaussian(randstellar(qpairbasis, 0), atol = 1e-5)
        @test !isgaussian(randstellar(qpairbasis, 1), atol = 1e-5)
    end

    @testset "metrics" begin
        rank = rand(0:3)
        x = randstellar(qpairbasis, rank)
        @test purity(x) == 1.0
        @test entropy_vn(x) == 0.0
        @test purity(fockstate(qpairbasis, 2)) == 1.0
        @test entropy_vn(fockstate(qpairbasis, 2)) == 0.0
    end
end