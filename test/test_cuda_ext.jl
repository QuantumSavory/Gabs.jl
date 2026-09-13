@testitem "CUDA extension" tags=[:cuda] begin
    using Gabs
    using CUDA
    using LinearAlgebra
    using Random

    # Every value below is compared against the same computation on `Array`s, so
    # a disagreement is a real difference and not a tolerance being met by luck.
    # `Float64` on both sides keeps the comparison about the code rather than
    # about precision. Scalar indexing is switched off for the whole item: a
    # method that falls back to indexing a `CuArray` element by element is a bug
    # even when it returns the right answer, and this turns that into a failure.
    CUDA.allowscalar(false)

    const CV = CuVector{Float64}
    const CM = CuMatrix{Float64}

    @test Base.get_extension(Gabs, :CUDAExt) !== nothing

    ongpu(x::GaussianState) = x.mean isa CuArray && x.covar isa CuArray
    ongpu(x::GaussianUnitary) = x.disp isa CuArray && x.symplectic isa CuArray
    ongpu(x::GaussianChannel) =
        x.disp isa CuArray && x.transform isa CuArray && x.noise isa CuArray

    # Compare a GPU object against its CPU counterpart field by field.
    function samemoments(cpu, gpu; atol = 1e-12)
        ongpu(gpu) || return false
        if cpu isa GaussianState
            return isapprox(Array(gpu.mean), cpu.mean; atol = atol) &&
                   isapprox(Array(gpu.covar), cpu.covar; atol = atol)
        elseif cpu isa GaussianUnitary
            return isapprox(Array(gpu.disp), cpu.disp; atol = atol) &&
                   isapprox(Array(gpu.symplectic), cpu.symplectic; atol = atol)
        else
            return isapprox(Array(gpu.disp), cpu.disp; atol = atol) &&
                   isapprox(Array(gpu.transform), cpu.transform; atol = atol) &&
                   isapprox(Array(gpu.noise), cpu.noise; atol = atol)
        end
    end

    for B in (QuadPairBasis, QuadBlockBasis)
        b1, b2, b3 = B(1), B(2), B(3)

        @testset "$(nameof(B)): predefined states" begin
            @test samemoments(vacuumstate(b1), vacuumstate(CV, CM, b1))
            @test samemoments(thermalstate(b1, 3), thermalstate(CV, CM, b1, 3))
            @test samemoments(coherentstate(b1, 1.0 + 0.5im), coherentstate(CV, CM, b1, 1.0 + 0.5im))
            @test samemoments(squeezedstate(b1, 0.3, 0.7), squeezedstate(CV, CM, b1, 0.3, 0.7))
            @test samemoments(eprstate(b2, 0.3, 0.7), eprstate(CV, CM, b2, 0.3, 0.7))
            # the single-type form has to reach both the vector and the matrix
            @test ongpu(vacuumstate(CuArray{Float64}, b2))
            # and the element type asked for is the one that comes back
            @test eltype(vacuumstate(CuVector{Float32}, CuMatrix{Float32}, b1).covar) === Float32
        end

        @testset "$(nameof(B)): predefined unitaries and channels" begin
            @test samemoments(displace(b1, 1.0 + 0.5im), displace(CV, CM, b1, 1.0 + 0.5im))
            @test samemoments(squeeze(b1, 0.3, 0.7), squeeze(CV, CM, b1, 0.3, 0.7))
            @test samemoments(twosqueeze(b2, 0.3, 0.7), twosqueeze(CV, CM, b2, 0.3, 0.7))
            @test samemoments(phaseshift(b1, 0.7), phaseshift(CV, CM, b1, 0.7))
            @test samemoments(beamsplitter(b2, 0.4), beamsplitter(CV, CM, b2, 0.4))
            @test samemoments(attenuator(b1, 0.5, 2), attenuator(CV, CM, b1, 0.5, 2))
            @test samemoments(amplifier(b1, 0.5, 2), amplifier(CV, CM, b1, 0.5, 2))
        end

        sc = squeezedstate(b1, 0.4, 0.9)
        sg = squeezedstate(CV, CM, b1, 0.4, 0.9)
        sc2 = coherentstate(b1, 0.7 - 0.2im)
        sg2 = coherentstate(CV, CM, b1, 0.7 - 0.2im)
        uc = displace(b1, 0.5 + 0.1im)
        ug = displace(CV, CM, b1, 0.5 + 0.1im)
        cc = attenuator(b1, 0.6, 1)
        cg = attenuator(CV, CM, b1, 0.6, 1)
        uc2 = beamsplitter(b2, 0.35)
        ug2 = beamsplitter(CV, CM, b2, 0.35)
        tc = sc ⊗ sc2 ⊗ sc
        tg = sg ⊗ sg2 ⊗ sg

        @testset "$(nameof(B)): operator algebra preserves the array type" begin
            @test samemoments(uc * sc, ug * sg)
            @test samemoments(cc * sc, cg * sg)
            @test samemoments(uc * uc, ug * ug)
            @test samemoments(apply!(copy(sc), uc), apply!(copy(sg), ug))
            @test samemoments(apply!(copy(sc), cc), apply!(copy(sg), cg))
            @test samemoments(apply!(copy(tc), [2], uc), apply!(copy(tg), [2], ug))
            @test samemoments(apply!(copy(tc), [2], cc), apply!(copy(tg), [2], cg))
            @test samemoments(apply!(copy(tc), [1, 2], uc2), apply!(copy(tg), [1, 2], ug2))
            @test samemoments(inv(uc), inv(ug))
        end

        @testset "$(nameof(B)): tensor, ptrace, embed, changebasis" begin
            @test samemoments(sc ⊗ sc2, sg ⊗ sg2)
            @test samemoments(uc ⊗ uc, ug ⊗ ug)
            @test samemoments(cc ⊗ cc, cg ⊗ cg)
            @test samemoments(tensor(Vector{Float64}, Matrix{Float64}, sc, sc2),
                              tensor(CV, CM, sg, sg2))
            @test samemoments(ptrace(tc, 2), ptrace(tg, 2))
            @test samemoments(ptrace(tc, [1, 3]), ptrace(tg, [1, 3]))
            @test samemoments(ptrace(Vector{Float64}, Matrix{Float64}, tc, 2),
                              ptrace(CV, CM, tg, 2))
            @test samemoments(embed(b3, 2, sc), embed(b3, 2, sg))
            @test samemoments(embed(b3, 2, uc), embed(b3, 2, ug))
            @test samemoments(embed(b3, 2, cc), embed(b3, 2, cg))
            other = B === QuadPairBasis ? QuadBlockBasis : QuadPairBasis
            @test samemoments(changebasis(other, tc), changebasis(other, tg))
            @test samemoments(changebasis(other, uc2), changebasis(other, ug2))
            @test samemoments(changebasis(other, cc ⊗ cc), changebasis(other, cg ⊗ cg))
            # a tensor product with one operand still on the host
            @test ongpu(sg ⊗ sc2)
            @test ongpu(sc ⊗ sg2)
        end

        @testset "$(nameof(B)): operands on different devices" begin
            # An operator and a state need not start on the same device. The
            # result belongs on the device, and has to equal the CPU answer.
            ref = uc * sc
            @test samemoments(ref, uc * sg)
            @test samemoments(ref, ug * sc)
            refc = cc * sc
            @test samemoments(refc, cc * sg)
            @test samemoments(refc, cg * sc)
            @test samemoments(ref, apply!(copy(sg), uc))
            @test samemoments(refc, apply!(copy(sg), cc))

            # and through a linear combination, which delegates per state
            lcc = GaussianLinearCombination(b1, [0.6, -0.8], [sc, sc2])
            lcg = GaussianLinearCombination(b1, [0.6, -0.8], [sg, sg2])
            @test samemoments((uc * lcc).states[1], (uc * lcg).states[1])
            @test samemoments((cc * lcc).states[1], (cc * lcg).states[1])
        end

        @testset "$(nameof(B)): metrics and phase space" begin
            @test purity(tg) ≈ purity(tc)
            @test entropy_vn(tg) ≈ entropy_vn(tc) atol = 1e-10
            @test fidelity(sg, sg2) ≈ fidelity(sc, sc2)
            @test sort(Array(sympspectrum(tg))) ≈ sort(sympspectrum(tc))
            @test logarithmic_negativity(eprstate(CV, CM, b2, 0.5, 0.3), 1) ≈
                  logarithmic_negativity(eprstate(b2, 0.5, 0.3), 1)
            @test issymplectic(b1, Array(ug.symplectic))
            # `isgaussian` compares eigenvalues against zero, and a pure state
            # sits exactly on that boundary; CUSOLVER and LAPACK land on
            # opposite sides of it by ~2e-16, so this needs a tolerance to be a
            # statement about the state rather than about the eigensolver.
            @test isgaussian(tg; atol = 1e-10) == isgaussian(tc; atol = 1e-10)

            x = [0.11, -0.23, 0.31, 0.07, -0.5, 0.19]
            @test wigner(tg, CuVector{Float64}(x)) ≈ wigner(tc, x)
            @test wignerchar(tg, CuVector{Float64}(x)) ≈ wignerchar(tc, x)
        end

        @testset "$(nameof(B)): batched phase-space evaluation" begin
            # A grid of points is the case worth sending to a device, and the
            # answer has to be the CPU's.
            xs = randn(6, 32)
            xg = CuMatrix{Float64}(xs)
            @test Array(wigner(tg, xg)) ≈ wigner(tc, xs)
            @test Array(wignerchar(tg, xg)) ≈ wignerchar(tc, xs)
            @test wigner(tg, xg) isa CuVector
            @test wignerchar(tg, xg) isa CuVector

            u1 = sc ⊗ sc2 ⊗ sc2
            v1 = sg ⊗ sg2 ⊗ sg2
            @test Array(cross_wigner(tg, v1, xg)) ≈ cross_wigner(tc, u1, xs)
            @test Array(cross_wignerchar(tg, v1, xg)) ≈ cross_wignerchar(tc, u1, xs)

            lcc = GaussianLinearCombination(tc.basis, [0.6, -0.8], [tc, u1])
            lcg = GaussianLinearCombination(tg.basis, [0.6, -0.8], [tg, v1])
            @test Array(wigner(lcg, xg)) ≈ wigner(lcc, xs)
            @test Array(wignerchar(lcg, xg)) ≈ wignerchar(lcc, xs)
            @test wigner(lcg, xg) isa CuVector

            # the single-point forms of the same functions, which reach the
            # interference sum and the cross terms through a different path
            x1 = xs[:, 1]
            xd = CuVector{Float64}(x1)
            @test cross_wigner(tg, v1, xd) ≈ cross_wigner(tc, u1, x1)
            @test cross_wignerchar(tg, v1, xd) ≈ cross_wignerchar(tc, u1, x1)
            @test wigner(lcg, xd) ≈ wigner(lcc, x1)
            @test wignerchar(lcg, xd) ≈ wignerchar(lcc, x1)
            # cross_wigner of a state with itself is its own Wigner function
            @test cross_wigner(tg, tg, xd) ≈ wigner(tg, xd)
            # and the pair is Hermitian
            @test cross_wigner(tg, v1, xd) ≈ conj(cross_wigner(v1, tg, xd))
        end

        @testset "$(nameof(B)): random objects" begin
            @test ongpu(randstate(CV, CM, b2))
            @test ongpu(randstate(CV, CM, b2; pure = true))
            @test ongpu(randunitary(CV, CM, b2))
            @test ongpu(randchannel(CV, CM, b2))
            @test randsymplectic(CM, b2) isa CuMatrix
            # drawn on the device, still a legitimate Gaussian object
            @test isgaussian(randstate(CV, CM, b2); atol = 1e-10)
            @test issymplectic(b2, Array(randunitary(CV, CM, b2).symplectic); atol = 1e-10)
        end

        @testset "$(nameof(B)): measurements" begin
            proj = Matrix{Float64}(I, 2, 2)
            Mc = generaldyne(tc, [2]; proj = proj)
            Mg = generaldyne(tg, [2]; proj = CuMatrix{Float64}(proj))
            @test ongpu(Mg.state)
            # the conditional covariance does not depend on the sampled outcome
            @test isapprox(Array(Mg.state.covar), Mc.state.covar; atol = 1e-10)
            # projecting onto a fixed state makes the whole result deterministic
            @test samemoments(generaldyne(tc, [2]; proj = vacuumstate(b1)).state,
                              generaldyne(tg, [2]; proj = vacuumstate(CV, CM, b1)).state;
                              atol = 1e-10)

            samples = rand(Generaldyne, tg, [2]; shots = 8)
            @test samples isa CuMatrix
            @test size(samples) == (2, 8)
            @test all(isfinite, Array(samples))

            Hg = homodyne(tg, [2], [0.0])
            @test ongpu(Hg.state)
            @test Hg.result isa CuVector
            hsamples = rand(Homodyne, tg, [2], [0.0]; shots = 8)
            @test hsamples isa CuMatrix
            @test size(hsamples) == (2, 8)
        end
    end

    @testset "Float32 round trip" begin
        b = QuadPairBasis(2)
        s32 = randstate(CuVector{Float32}, CuMatrix{Float32}, b)
        @test eltype(s32.mean) === Float32
        @test eltype(s32.covar) === Float32
        u32 = displace(CuVector{Float32}, CuMatrix{Float32}, b, 0.3f0 + 0.2f0im)
        out = u32 * s32
        @test eltype(out.covar) === Float32
        @test ongpu(out)
        # the same computation in double precision on the host, to the accuracy
        # single precision can carry
        s64 = GaussianState(b, Array(Float64.(s32.mean)), Array(Float64.(s32.covar)); ħ = s32.ħ)
        u64 = displace(b, 0.3 + 0.2im)
        @test isapprox(Array(out.covar), (u64 * s64).covar; rtol = 1e-5)
    end
end
