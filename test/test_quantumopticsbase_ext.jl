@testitem "QuantumOpticsBase extension" begin
    using Gabs
    using LinearAlgebra
    using QuantumInterface: QuantumOpticsRepr, express
    using QuantumOpticsBase

    const QO = QuantumOpticsBase

    function samestate(ket1, ket2; atol=1e-10)
        return isapprox(QO.dm(ket1), QO.dm(ket2); atol=atol, rtol=0)
    end

    @test Base.get_extension(Gabs, :QuantumOpticsBaseExt) !== nothing

    # This helper keeps only the creation-operator part of the squeezed-vacuum
    # expansion. It matches the Gaussian-to-Ket construction more closely in a
    # truncated Fock basis than `QO.squeeze`, which exponentiates truncated
    # creation/annihilation operators directly.
    function manual_squeezed_state(basis, z)
        λ = -tanh(abs(z)) * exp(im * angle(z))
        return normalize!(exp(QO.dense((λ / 2) * QO.create(basis)^2)) * QO.fockstate(basis, 0))
    end

    # This is the displaced analogue of `manual_squeezed_state`: it rewrites the
    # displaced squeezed vacuum as a creation-only exponential acting on vacuum.
    # `QO.displace * QO.squeeze` is still tested below, but its finite-dimensional
    # matrix exponentials are a bit more sensitive to cutoff effects.
    function manual_displaced_squeezed_state(basis, α, z)
        λ = -tanh(abs(z)) * exp(im * angle(z))
        β = α - λ * conj(α)
        generator = QO.dense((λ / 2) * QO.create(basis)^2 + β * QO.create(basis))
        return normalize!(exp(generator) * QO.fockstate(basis, 0))
    end

    function evolve_with_generator(generator, ket)
        return normalize!(exp(QO.dense(generator)) * ket)
    end

    function two_mode_ops(cutoff)
        mode_basis = QO.FockBasis(cutoff)
        basis = QO.tensor(mode_basis, mode_basis)
        a1 = QO.embed(basis, 1, QO.destroy(mode_basis))
        a2 = QO.embed(basis, 2, QO.destroy(mode_basis))
        ad1 = QO.embed(basis, 1, QO.create(mode_basis))
        ad2 = QO.embed(basis, 2, QO.create(mode_basis))
        n1 = QO.embed(basis, 1, QO.number(mode_basis))
        n2 = QO.embed(basis, 2, QO.number(mode_basis))
        return (; basis, mode_basis, a1, a2, ad1, ad2, n1, n2)
    end

    @testset "single-mode states" begin
        repr = QuantumOpticsRepr(8)
        basis = QO.FockBasis(repr.cutoff)
        vacuum = QO.fockstate(basis, 0)
        builtin_squeeze_atol = 1e-3
        builtin_displaced_squeeze_atol = 1e-2

        @test samestate(express(Gabs.vacuumstate(Gabs.QuadPairBasis(1)), repr), vacuum)
        @test samestate(express(Gabs.vacuumstate(Gabs.QuadBlockBasis(1)), repr), vacuum)

        α = 0.3 - 0.4im
        @test samestate(
            express(Gabs.coherentstate(Gabs.QuadPairBasis(1), α), repr),
            QO.coherentstate(basis, α),
        )
        @test samestate(
            express(Gabs.coherentstate(Gabs.QuadBlockBasis(1), α), repr),
            QO.coherentstate(basis, α),
        )

        z = 0.35 * exp(0.4im)
        squeezed = manual_squeezed_state(basis, z)
        builtin_squeezed = QO.squeeze(basis, z) * vacuum
        @test samestate(
            express(Gabs.squeezedstate(Gabs.QuadPairBasis(1), abs(z), angle(z)), repr),
            squeezed,
        )
        @test samestate(
            express(Gabs.squeezedstate(Gabs.QuadPairBasis(1), abs(z), angle(z)), repr),
            builtin_squeezed;
            atol=builtin_squeeze_atol,
        )
        @test samestate(
            express(Gabs.squeezedstate(Gabs.QuadBlockBasis(1), abs(z), angle(z)), repr),
            squeezed,
        )
        @test samestate(
            express(Gabs.squeezedstate(Gabs.QuadBlockBasis(1), abs(z), angle(z)), repr),
            builtin_squeezed;
            atol=builtin_squeeze_atol,
        )

        displaced = manual_displaced_squeezed_state(basis, α, z)
        builtin_displaced = QO.displace(basis, α) * QO.squeeze(basis, z) * vacuum
        state_pair = Gabs.displace(Gabs.QuadPairBasis(1), α) *
            Gabs.squeezedstate(Gabs.QuadPairBasis(1), abs(z), angle(z))
        state_block = Gabs.displace(Gabs.QuadBlockBasis(1), α) *
            Gabs.squeezedstate(Gabs.QuadBlockBasis(1), abs(z), angle(z))
        @test samestate(express(state_pair, repr), displaced)
        @test samestate(express(state_block, repr), displaced)
        @test samestate(express(state_pair, repr), builtin_displaced; atol=builtin_displaced_squeeze_atol)
        @test samestate(express(state_block, repr), builtin_displaced; atol=builtin_displaced_squeeze_atol)
    end

    @testset "multi-mode states" begin
        repr = QuantumOpticsRepr(6)
        mode_basis = QO.FockBasis(repr.cutoff)
        basis = QO.tensor(mode_basis, mode_basis)
        vacuum = QO.tensor(QO.fockstate(mode_basis, 0), QO.fockstate(mode_basis, 0))
        builtin_squeeze_atol = 2e-3

        α = [0.2 + 0.1im, -0.15 + 0.3im]
        expected = QO.tensor(QO.coherentstate(mode_basis, α[1]), QO.coherentstate(mode_basis, α[2]))
        @test samestate(express(Gabs.coherentstate(Gabs.QuadPairBasis(2), α), repr), expected)
        @test samestate(express(Gabs.coherentstate(Gabs.QuadBlockBasis(2), α), repr), expected)

        z = [-0.25im, 0.3]
        squeezed = QO.tensor(
            manual_squeezed_state(mode_basis, z[1]),
            manual_squeezed_state(mode_basis, z[2]),
        )
        builtin_squeezed = QO.tensor(
            QO.squeeze(mode_basis, z[1]) * QO.fockstate(mode_basis, 0),
            QO.squeeze(mode_basis, z[2]) * QO.fockstate(mode_basis, 0),
        )
        @test samestate(
            express(Gabs.squeezedstate(Gabs.QuadPairBasis(2), abs.(z), angle.(z)), repr),
            squeezed;
            atol=1e-9,
        )
        @test samestate(
            express(Gabs.squeezedstate(Gabs.QuadPairBasis(2), abs.(z), angle.(z)), repr),
            builtin_squeezed;
            atol=builtin_squeeze_atol,
        )
        @test samestate(
            express(Gabs.squeezedstate(Gabs.QuadBlockBasis(2), abs.(z), angle.(z)), repr),
            squeezed;
            atol=1e-9,
        )
        @test samestate(
            express(Gabs.squeezedstate(Gabs.QuadBlockBasis(2), abs.(z), angle.(z)), repr),
            builtin_squeezed;
            atol=builtin_squeeze_atol,
        )

        create1 = QO.embed(basis, 1, QO.create(mode_basis))
        create2 = QO.embed(basis, 2, QO.create(mode_basis))
        λ = -tanh(0.27)
        epr = normalize!(exp(QO.dense(λ * create1 * create2)) * vacuum)
        @test samestate(express(Gabs.eprstate(Gabs.QuadPairBasis(2), 0.27, 0.0), repr), epr; atol=1e-8)
        @test samestate(express(Gabs.eprstate(Gabs.QuadBlockBasis(2), 0.27, 0.0), repr), epr; atol=1e-8)
    end

    @testset "operator actions" begin
        repr = QuantumOpticsRepr(12)
        ops = two_mode_ops(repr.cutoff)
        α0 = [0.12 + 0.03im, -0.07 + 0.08im]
        pair_state = Gabs.coherentstate(Gabs.QuadPairBasis(2), α0)
        block_state = Gabs.coherentstate(Gabs.QuadBlockBasis(2), α0)

        @test samestate(express(pair_state, repr), express(block_state, repr))

        function check_operator_action(pair_state, block_state, pair_op, block_op, generator; atol=1e-8)
            quantumoptics_state = express(pair_state, repr)
            qostate_after_op = evolve_with_generator(generator, quantumoptics_state)
            @test samestate(express(pair_op * pair_state, repr), qostate_after_op; atol=atol)
            @test samestate(express(block_op * block_state, repr), qostate_after_op; atol=atol)
        end

        transmit = 0.2
        beamsplitter_generator = asin(sqrt(transmit)) * (ops.ad1 * ops.a2 - ops.a1 * ops.ad2)
        check_operator_action(
            pair_state,
            block_state,
            Gabs.beamsplitter(Gabs.QuadPairBasis(2), transmit),
            Gabs.beamsplitter(Gabs.QuadBlockBasis(2), transmit),
            beamsplitter_generator,
        )

        phases = [0.17, -0.21]
        phaseshift_generator = -im * (phases[1] * ops.n1 + phases[2] * ops.n2)
        check_operator_action(
            pair_state,
            block_state,
            Gabs.phaseshift(Gabs.QuadPairBasis(2), phases),
            Gabs.phaseshift(Gabs.QuadBlockBasis(2), phases),
            phaseshift_generator,
        )

        displacement = [0.04 - 0.01im, -0.03 + 0.02im]
        displacement_generator = displacement[1] * ops.ad1 - conj(displacement[1]) * ops.a1 +
            displacement[2] * ops.ad2 - conj(displacement[2]) * ops.a2
        check_operator_action(
            pair_state,
            block_state,
            Gabs.displace(Gabs.QuadPairBasis(2), displacement),
            Gabs.displace(Gabs.QuadBlockBasis(2), displacement),
            displacement_generator,
        )

        squeeze_parameters = [0.05 * exp(0.2im), -0.04im]
        squeeze_generator = conj(squeeze_parameters[1]) / 2 * ops.a1^2 -
            squeeze_parameters[1] / 2 * ops.ad1^2 +
            conj(squeeze_parameters[2]) / 2 * ops.a2^2 -
            squeeze_parameters[2] / 2 * ops.ad2^2
        check_operator_action(
            pair_state,
            block_state,
            Gabs.squeeze(Gabs.QuadPairBasis(2), abs.(squeeze_parameters), angle.(squeeze_parameters)),
            Gabs.squeeze(Gabs.QuadBlockBasis(2), abs.(squeeze_parameters), angle.(squeeze_parameters)),
            squeeze_generator,
        )

        twosqueeze_r = 0.05
        twosqueeze_phase = -0.3
        twosqueeze_generator = -(twosqueeze_r * exp(im * twosqueeze_phase)) * ops.ad1 * ops.ad2 +
            (twosqueeze_r * exp(-im * twosqueeze_phase)) * ops.a1 * ops.a2
        check_operator_action(
            pair_state,
            block_state,
            Gabs.twosqueeze(Gabs.QuadPairBasis(2), twosqueeze_r, twosqueeze_phase),
            Gabs.twosqueeze(Gabs.QuadBlockBasis(2), twosqueeze_r, twosqueeze_phase),
            twosqueeze_generator,
        )
    end

    @testset "mixed states are rejected" begin
        repr = QuantumOpticsRepr(6)
        @test_throws ArgumentError express(Gabs.thermalstate(Gabs.QuadPairBasis(1), 0.5), repr)
        @test_throws ArgumentError express(Gabs.thermalstate(Gabs.QuadBlockBasis(1), 0.5), repr)
    end
end
