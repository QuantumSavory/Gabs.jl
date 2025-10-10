@testitem "Measurements" begin
    using Gabs
    using StableRNGs
    using StaticArrays
    using LinearAlgebra: det, I, cholesky, Symmetric

    @testset "generaldyne" begin

        qpairbasis, qblockbasis = QuadPairBasis(1), QuadBlockBasis(1)
        for basis in [qpairbasis, qblockbasis]
            vac = vacuumstate(basis)
            @test_throws ArgumentError homodyne(vac, [1, 2])
            @test_throws ArgumentError generaldyne(vac ⊗ vac, [1], proj = zeros(4, 4))
            @test_throws ArgumentError generaldyne(vac ⊗ vac, [1], proj = GaussianState(basis ⊕ basis, zeros(4), zeros(4, 4)))
            vacs = vac ⊗ vac ⊗ vac ⊗ vac
            gd1 = generaldyne(vacs, [2, 4], proj = vac ⊗ vac)
            @test isapprox(gd1.result, vac ⊗ vac, atol = 1e-12)
            @test isapprox(gd1.state, vacs, atol = 1e-12)
            result, state = gd1
            @test result == gd1.result && state == gd1.state

            coh = coherentstate(basis, 1.0+im)
            cohs = coh ⊗ vac ⊗ coh ⊗ vac
            epr = eprstate(basis ⊕ basis, 1.0, 3.0)
            gd2 = generaldyne(cohs, [1, 4], proj = epr)
            @test isapprox(gd2.result, epr, atol = 1e-12)
            @test isapprox(gd2.state, vac ⊗ vac ⊗ coh ⊗ vac, atol = 1e-12)
        end

        indices, nmodes = [7, 8, 9, 10], 10

        # random Gaussian state tests
        rs_qpair = randstate(QuadPairBasis(nmodes))
        rs_qblock = changebasis(QuadBlockBasis, rs_qpair)
        proj_qpair = randstate(QuadPairBasis(length(indices)))
        proj_qblock = changebasis(QuadBlockBasis, proj_qpair)
        gd3_qpair = generaldyne(rs_qpair, indices, proj = proj_qpair)
        gd3_qblock = generaldyne(rs_qblock, indices, proj = proj_qblock)

        # analytical calculation of evolved subsystem that is not measured
        xA, xB, VA, VB, VAB = Gabs._part_state(rs_qpair, indices)
        gd3_evolved_mean = xA .+ VAB * ((inv(VB .+ proj_qpair.covar)) * (proj_qpair.mean .- xB))
        gd3_evolved_covar = VA .- VAB * ((inv(VB .+ proj_qpair.covar)) * transpose(VAB))
        out3_mean = vcat(gd3_evolved_mean, zeros(2*length(indices)))
        out3_covar = zeros(2*nmodes, 2*nmodes)
        copyto!(@view(out3_covar[1:2*(nmodes-length(indices)), 1:2*(nmodes-length(indices))]), gd3_evolved_covar)
        copyto!(@view(out3_covar[2*(nmodes-length(indices))+1:2*nmodes, 2*(nmodes-length(indices))+1:2*nmodes]), Matrix{Float64}(I, 2*length(indices), 2*length(indices)))

        evolved_state_qpair = GaussianState(QuadPairBasis(nmodes), out3_mean, out3_covar)
        evolved_state_qblock = changebasis(QuadBlockBasis, evolved_state_qpair)
        @test isapprox(gd3_qpair.state, evolved_state_qpair)
        @test isapprox(gd3_qblock.state, evolved_state_qblock)
        
        # tests to check that static arrays are outputted for generaldyne 
        # measurements of Gaussian states wrapping static arrays
        sstatic = vacuumstate(SVector{2}, SMatrix{2,2}, QuadPairBasis(1))
        statestatic = sstatic ⊗ sstatic ⊗ sstatic ⊗ sstatic
        gdstatic = generaldyne(statestatic, [2])
        @test (gdstatic.state).mean isa SVector && (gdstatic.state).covar isa SMatrix
        @test isequal(gdstatic.state, statestatic)

        # TODO: add more tests for random Generaldyne sampling
        @test_throws ArgumentError rand(Generaldyne, rs_qpair, collect(1:11))
        @test_throws ArgumentError rand(Generaldyne, rs_qblock, collect(1:11))
        @test_throws ArgumentError rand(Generaldyne, rs_qpair, indices, proj = zeros(20, 20))
        @test_throws ArgumentError rand(Generaldyne, rs_qblock, indices, proj = zeros(20, 20))
        @test size(rand(Generaldyne, rs_qpair, [1, 3, 5], shots = 10)) == (6, 10)
        @test size(rand(Generaldyne, rs_qblock, [1, 3, 5], shots = 10)) == (6, 10)

    end

    @testset "homodyne" begin

        @testset "homodyne" begin
            qpairbasis, qblockbasis = QuadPairBasis(1), QuadBlockBasis(1)
        
            for basis in (qpairbasis, qblockbasis)
                vac = vacuumstate(basis)
                @test_throws MethodError homodyne(vac, [1, 2], [0.0])
                @test_throws MethodError homodyne(vac, [1], [0.0, π/2])
            end
        
            # simple test case: measuring 1 mode of 2-mode state
            for basis in (QuadPairBasis(2), QuadBlockBasis(2))
                st = vacuumstate(basis) ⊗ vacuumstate(basis)
                M = homodyne(st, [1], [0.0])
                @test isa(M, Generaldyne)
                @test M.result isa Vector
                @test M.state isa GaussianState
        
                result, state = M
                @test result == M.result
                @test state == M.state
        
                # check if measured mode is replaced with vacuum
                @test isapprox(state.mean[1:2], zeros(2), atol=1e-12)
                @test isapprox(state.covar[1:2, 1:2], Matrix{Float64}(I,2,2), atol=1e-12)
            end
        
            rng = StableRNG(123)
            coh = coherentstate(QuadPairBasis(3), 1.0)
            coh_block = changebasis(QuadBlockBasis, coh)
            angles, indices = [0.0], [2]
            Random.seed!(rng, 123)
            M_pair = homodyne(coh, indices, angles)
            Random.seed!(rng, 123)
            M_block = homodyne(coh_block, indices, angles)
            @test isapprox(M_pair.state, changebasis(QuadPairBasis, M_block.state))
            @test isapprox(M_pair.result, M_block.result, atol=1e-12)
        
            st = squeezedstate(QuadPairBasis(4), 0.5, π/2)
            st_block = changebasis(QuadBlockBasis, st)
            indices = [2, 4]
            angles = [0.0, π/2]
            @test size(rand(Homodyne, st, indices, angles, shots=10)) == (4, 10)
            @test size(rand(Homodyne, st_block, indices, angles, shots=7)) == (4, 7)
        
            indices = [1, 2]
            rs_qpair = randstate(QuadPairBasis(4))
            rs_qblock = changebasis(QuadBlockBasis, rs_qpair)
            M_qpair = homodyne(rs_qpair, indices, [0.0, π/2])
            M_qblock = homodyne(rs_qblock, indices, [0.0, π/2])
        
            # extract analytical conditional update
            xA, xB, VA, VB, VAB = Gabs._part_state(rs_qpair, indices)
            B = copy(VB)
            # apply infinite squeezing (0.0 and π/2)
            θs = [0.0, π/2]
            for i in 1:2
                θ = θs[i]
                sq = 1e-12
                ct, st = cos(θ), sin(θ)
                B[2i-1,2i-1] += ct^2 * sq + st^2 / sq
                B[2i-1,2i]   += ct * st * (sq - 1 / sq)
                B[2i,2i-1]   += ct * st * (sq - 1 / sq)
                B[2i,2i]     += st^2 * sq + ct^2 / sq
            end
            L = cholesky(Symmetric(B)).L
            resultmean = L * randn(4) + xB
            xA′ = xA .+ VAB * (inv(B) * (resultmean - xB))
            VA′ = VA .- VAB * (inv(B) * transpose(VAB))
        
            nm = 4
            out_mean = zeros(2nm)
            notindices = setdiff(1:nm, indices)
            for i in eachindex(notindices)
                idx = notindices[i]
                out_mean[2idx-1:2idx] .= @view(xA′[2i-1:2i])
            end
            out_covar = Matrix{Float64}(I, 2nm, 2nm)
            for i in eachindex(notindices), j in i:length(notindices)
                idx, jdx = notindices[i], notindices[j]
                out_covar[2idx-1:2idx, 2jdx-1:2jdx] .= @view(VA′[2i-1:2i, 2j-1:2j])
                out_covar[2jdx-1:2jdx, 2idx-1:2idx] .= @view(VA′[2j-1:2j, 2i-1:2i])
            end
            expected_state = GaussianState(QuadPairBasis(nm), out_mean, out_covar)
            @test isapprox(M_qpair.state, expected_state, atol=1e-8)
            @test isapprox(M_qblock.state, changebasis(QuadBlockBasis, expected_state), atol=1e-8)
        
            sstatic = vacuumstate(SVector{2}, SMatrix{2,2}, QuadPairBasis(1))
            statestatic = sstatic ⊗ sstatic ⊗ sstatic ⊗ sstatic
            hstatic = homodyne(statestatic, [2], [π/4])
            @test (hstatic.state).mean isa SVector && (hstatic.state).covar isa SMatrix
            @test isequal(hstatic.state.mean[1:2], zeros(2))

            @test_throws ArgumentError rand(Homodyne, rs_qpair, collect(1:5), [0.0])
            @test_throws ArgumentError rand(Homodyne, rs_qblock, collect(1:5), [π/2])
        end        
    end
end