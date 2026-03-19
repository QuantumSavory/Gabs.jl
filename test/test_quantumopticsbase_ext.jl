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

    function manual_squeezed_state(basis, z)
        λ = -tanh(abs(z)) * exp(im * angle(z))
        return normalize!(exp(QO.dense((λ / 2) * QO.create(basis)^2)) * QO.fockstate(basis, 0))
    end

    function manual_displaced_squeezed_state(basis, α, z)
        λ = -tanh(abs(z)) * exp(im * angle(z))
        β = α - λ * conj(α)
        generator = QO.dense((λ / 2) * QO.create(basis)^2 + β * QO.create(basis))
        return normalize!(exp(generator) * QO.fockstate(basis, 0))
    end

    @testset "single-mode states" begin
        repr = QuantumOpticsRepr(8)
        basis = QO.FockBasis(repr.cutoff)
        vacuum = QO.fockstate(basis, 0)

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
        @test samestate(
            express(Gabs.squeezedstate(Gabs.QuadPairBasis(1), abs(z), angle(z)), repr),
            squeezed,
        )
        @test samestate(
            express(Gabs.squeezedstate(Gabs.QuadBlockBasis(1), abs(z), angle(z)), repr),
            squeezed,
        )

        displaced = manual_displaced_squeezed_state(basis, α, z)
        state_pair = Gabs.displace(Gabs.QuadPairBasis(1), α) *
            Gabs.squeezedstate(Gabs.QuadPairBasis(1), abs(z), angle(z))
        state_block = Gabs.displace(Gabs.QuadBlockBasis(1), α) *
            Gabs.squeezedstate(Gabs.QuadBlockBasis(1), abs(z), angle(z))
        @test samestate(express(state_pair, repr), displaced)
        @test samestate(express(state_block, repr), displaced)
    end

    @testset "multi-mode states" begin
        repr = QuantumOpticsRepr(6)
        mode_basis = QO.FockBasis(repr.cutoff)
        basis = QO.tensor(mode_basis, mode_basis)
        vacuum = QO.tensor(QO.fockstate(mode_basis, 0), QO.fockstate(mode_basis, 0))

        α = [0.2 + 0.1im, -0.15 + 0.3im]
        expected = QO.tensor(QO.coherentstate(mode_basis, α[1]), QO.coherentstate(mode_basis, α[2]))
        @test samestate(express(Gabs.coherentstate(Gabs.QuadPairBasis(2), α), repr), expected)
        @test samestate(express(Gabs.coherentstate(Gabs.QuadBlockBasis(2), α), repr), expected)

        z = [-0.25im, 0.3]
        squeezed = QO.tensor(
            manual_squeezed_state(mode_basis, z[1]),
            manual_squeezed_state(mode_basis, z[2]),
        )
        @test samestate(
            express(Gabs.squeezedstate(Gabs.QuadPairBasis(2), abs.(z), angle.(z)), repr),
            squeezed;
            atol=1e-9,
        )
        @test samestate(
            express(Gabs.squeezedstate(Gabs.QuadBlockBasis(2), abs.(z), angle.(z)), repr),
            squeezed;
            atol=1e-9,
        )

        create1 = QO.embed(basis, 1, QO.create(mode_basis))
        create2 = QO.embed(basis, 2, QO.create(mode_basis))
        λ = -tanh(0.27)
        epr = normalize!(exp(QO.dense(λ * create1 * create2)) * vacuum)
        @test samestate(express(Gabs.eprstate(Gabs.QuadPairBasis(2), 0.27, 0.0), repr), epr; atol=1e-8)
        @test samestate(express(Gabs.eprstate(Gabs.QuadBlockBasis(2), 0.27, 0.0), repr), epr; atol=1e-8)
    end

    @testset "mixed states are rejected" begin
        repr = QuantumOpticsRepr(6)
        @test_throws ArgumentError express(Gabs.thermalstate(Gabs.QuadPairBasis(1), 0.5), repr)
        @test_throws ArgumentError express(Gabs.thermalstate(Gabs.QuadBlockBasis(1), 0.5), repr)
    end
end
