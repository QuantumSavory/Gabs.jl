@testitem "Measurements" begin
    using Gabs
    using StaticArrays
    using LinearAlgebra: det, I

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

        qpairbasis, qblockbasis = QuadPairBasis(1), QuadBlockBasis(1)
        for basis in [qpairbasis, qblockbasis]
            vac = vacuumstate(basis)
            @test_throws ArgumentError homodyne(vac, [1, 2], [0.0, pi])
            @test_throws ArgumentError homodyne(vac ⊗ vac, [1], [0.0, pi])
            vacs = vac ⊗ vac ⊗ vac ⊗ vac
            hd1 = homodyne(vacs, [2, 4], [0.0, pi/2])
            @test isapprox(hd1.result, vac ⊗ vac, atol = 1e-12)
            @test isapprox(hd1.state, vacs, atol = 1e-12)
            result, state = hd1
            @test result == hd1.result && state == hd1.state

            coh = coherentstate(basis, 1.0+im)
            cohs = coh ⊗ vac ⊗ coh ⊗ vac
            epr = eprstate(basis ⊕ basis, 1.0, 3.0)
            hd2 = homodyne(cohs, [1, 4], [0.0, pi/4])
            @test isapprox(hd2.result, epr, atol = 1e-12)
            @test isapprox(hd2.state, vac ⊗ vac ⊗ coh ⊗ vac, atol = 1e-12)
        end

        indices, angles, nmodes = [7, 8, 9, 10], [pi/2, pi/4, pi/3, pi/6], 10

        # random Gaussian state tests
        rs_qpair = randstate(QuadPairBasis(nmodes))
        rs_qblock = changebasis(QuadBlockBasis, rs_qpair)
        proj_qpair = randstate(QuadPairBasis(length(indices)))
        proj_qblock = changebasis(QuadBlockBasis, proj_qpair)
        gd3_qpair = homodyne(rs_qpair, indices, angles)
        gd3_qblock = homodyne(rs_qblock, indices, angles)

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
        gdstatic = homodyne(statestatic, [2])
        @test (gdstatic.state).mean isa SVector && (gdstatic.state).covar isa SMatrix
        @test isequal(gdstatic.state, statestatic)

        # TODO: add more tests for random Homodyne sampling
        @test_throws ArgumentError rand(Homodyne, rs_qpair, collect(1:11))
        @test_throws ArgumentError rand(Homodyne, rs_qblock, collect(1:11))
        @test_throws ArgumentError rand(Homodyne, rs_qpair, indices, proj = zeros(20, 20))
        @test_throws ArgumentError rand(Homodyne, rs_qblock, indices, proj = zeros(20, 20))
        @test size(rand(Homodyne, rs_qpair, [1, 3, 5], [pi/2, pi/3, pi/4], shots = 10)) == (6, 10)
        @test size(rand(Homodyne, rs_qblock, [1, 3, 5],[pi/2, pi/3, pi/4],  shots = 10)) == (6, 10)
    end
end