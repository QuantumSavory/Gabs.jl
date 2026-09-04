@testitem "MPS" begin
    using Gabs
    using ITensors
    using ITensorMPS
    using LinearAlgebra
    using Random

    ext = Base.get_extension(Gabs, :ITensorMPSExt)

    function densevector(psi::MPS)
        sites = siteinds(psi)
        T = ITensorMPS.contract(psi)
        return Array(T, sites...)
    end

    @testset "mpsstate requires the extension" begin
        # this testitem itself loads ITensors/ITensorMPS, so this just checks that the
        # fallback error type/message is the one Gabs defines, not that it fires here.
        @test occursin("ITensors.jl and ITensorMPS.jl", Gabs.MPS_EXT_ERROR)
    end

    @testset "_reck_decompose round-trip" begin
        Random.seed!(42)
        for N in 1:5, _ in 1:3
            if N == 1
                U = reshape([cis(2π * rand())], 1, 1)
            else
                A = randn(ComplexF64, N, N)
                Q, R = qr(A)
                U = Matrix(Q) * Diagonal(sign.(diag(R)))
            end
            gates, phases = ext._reck_decompose(U)
            result = Matrix{ComplexF64}(I, N, N)
            for (c, θ, φ) in gates
                result = ext._embed_givens(N, c, θ, φ) * result
            end
            result = Diagonal(cis.(phases)) * result
            @test isapprox(result, U; atol = 1e-9)
        end
    end

    @testset "single-mode analytic Fock coefficients" begin
        basis = QuadPairBasis(1)

        @testset "vacuum" begin
            psi = mpsstate(vacuumstate(basis); cutoff = 8)
            v = densevector(psi)
            expected = zeros(ComplexF64, 8)
            expected[1] = 1.0
            # `mpsstate` regularizes near-degenerate Bloch-Messiah inputs with a tiny
            # (O(1e-6)) symplectic perturbation (see `_regularize_symplectic`), so an exact
            # vacuum picks up a correspondingly tiny population above the vacuum Fock level.
            @test isapprox(v, expected; atol = 1e-5)
        end

        @testset "coherent" begin
            alpha = 0.6 + 0.3im
            cutoff = 24
            psi = mpsstate(coherentstate(basis, alpha); cutoff = cutoff)
            v = densevector(psi)
            expected = [exp(-abs2(alpha)/2) * alpha^n / sqrt(factorial(big(n))) for n in 0:cutoff-1]
            expected = ComplexF64.(expected)
            # global phase is physically irrelevant; align before comparing
            v = v .* cis(-angle(v[1]))
            expected = expected .* cis(-angle(expected[1]))
            @test isapprox(v, expected; atol = 1e-6)
        end

        @testset "squeezed vacuum, convergence with cutoff" begin
            r = 0.7
            errs = Float64[]
            for cutoff in (10, 20, 30)
                psi = mpsstate(squeezedstate(basis, r, 0.0); cutoff = cutoff)
                v = densevector(psi)
                expected = zeros(ComplexF64, cutoff)
                for n in 0:2:cutoff-1
                    k = n ÷ 2
                    expected[n+1] = (1/sqrt(cosh(r))) * (-tanh(r))^k * sqrt(factorial(big(n))) / (2.0^k * factorial(big(k)))
                end
                v = v .* cis(-angle(v[1]))
                expected = expected .* cis(-angle(expected[1]))
                push!(errs, norm(v[1:min(10,cutoff)] .- expected[1:min(10,cutoff)]))
            end
            @test errs[end] <= errs[1]
            @test errs[end] < 1e-6
        end

        @testset "thermal state is rejected (mixed)" begin
            @test_throws ErrorException mpsstate(thermalstate(basis, 1); cutoff = 8)
        end
    end

    @testset "two-mode product state moments" begin
        # a genuine tensor product of two differently-squeezed single-mode states: the
        # target covariance is block-diagonal, giving a strong, independent end-to-end check
        # of the full williamson -> blochmessiah -> Reck-mesh -> gate-application pipeline
        # (as opposed to the single-mode tests, which never exercise the two-mode gates).
        basis1 = QuadPairBasis(1)
        state = squeezedstate(basis1, 0.5, 0.0) ⊗ squeezedstate(basis1, 0.3, 0.7)
        cutoff = 20
        psi = mpsstate(state; cutoff = cutoff)
        sites = siteinds(psi)
        v = vec(Array(ITensorMPS.contract(psi), sites...))
        a = zeros(ComplexF64, cutoff, cutoff)
        for k in 1:cutoff-1
            a[k, k+1] = sqrt(k)
        end
        Id = Matrix{ComplexF64}(I, cutoff, cutoff)
        a1, a2 = kron(Id, a), kron(a, Id)
        ħ = state.ħ
        mean = zeros(4)
        for (i, ai) in enumerate((a1, a2))
            eai = v' * ai * v
            mean[2i-1] = sqrt(2ħ) * real(eai)
            mean[2i] = sqrt(2ħ) * imag(eai)
        end
        covar = zeros(4, 4)
        ops = (a1, a2)
        for i in 1:2, j in 1:2
            ai, aj = ops[i], ops[j]
            eaiaj, eaidagaj = v' * (ai*aj) * v, v' * (ai'*aj) * v
            eaiajdag, eaidagajdag = v' * (ai*aj') * v, v' * (ai'*aj') * v
            mxi, mpi = mean[2i-1], mean[2i]
            mxj, mpj = mean[2j-1], mean[2j]
            covar[2i-1, 2j-1] = real((ħ/2) * (eaiaj + eaidagajdag + eaiajdag + eaidagaj)) - mxi*mxj
            covar[2i, 2j] = real(-(ħ/2) * (eaiaj + eaidagajdag - eaiajdag - eaidagaj)) - mpi*mpj
            covar[2i-1, 2j] = real(-im*(ħ/2) * (eaiaj - eaidagajdag - eaiajdag + eaidagaj)) - mxi*mpj
            covar[2i, 2j-1] = real(-im*(ħ/2) * (eaiaj - eaidagajdag + eaiajdag - eaidagaj)) - mpi*mxj
        end
        @test isapprox(mean, state.mean; atol = 1e-4)
        @test isapprox(covar, state.covar; atol = 1e-4)
    end

    @testset "entangled multi-mode states report a clear blochmessiah error" begin
        # `SymplecticMatrices.jl`'s `blochmessiah` has been observed to occasionally return
        # a passive transformation that is not itself symplectic for certain multi-mode
        # states reached via `williamson` (e.g. the EPR/TMSV state below); `mpsstate` must
        # detect this and fail loudly rather than silently return a wrong state.
        basis2 = QuadPairBasis(2)
        @test_throws ErrorException mpsstate(eprstate(basis2, 0.6, 0.4); cutoff = 10)
    end

    @testset "norm sanity" begin
        basis = QuadPairBasis(1)
        for state in (
            vacuumstate(basis),
            coherentstate(basis, 1.1 - 0.4im),
            squeezedstate(basis, 0.5, 1.0),
        )
            psi = mpsstate(state; cutoff = 20)
            @test isapprox(norm(psi), 1.0; atol = 1e-6)
            @test isapprox(inner(psi, psi), 1.0; atol = 1e-6)
        end
    end

    @testset "QuadBlockBasis input" begin
        basis = QuadBlockBasis(1)
        alpha = 0.5 + 0.2im
        psi_block = mpsstate(coherentstate(basis, alpha); cutoff = 16)
        psi_pair = mpsstate(coherentstate(QuadPairBasis(1), alpha); cutoff = 16)
        @test isapprox(abs(inner(psi_block, psi_pair)), 1.0; atol = 1e-6)
    end
end
