@testitem "Linear Combinations" begin
    using Gabs
    using StaticArrays
    using LinearAlgebra

    nmodes = rand(1:3)
    qpairbasis = QuadPairBasis(nmodes)
    qblockbasis = QuadBlockBasis(nmodes)

    @testset "Basic constructors" begin
        vac = vacuumstate(qpairbasis)
        lc1 = GaussianLinearCombination(vac)
        @test lc1 isa GaussianLinearCombination

        @test length(lc1) == 1
        @test lc1[1][1] == 1.0
        @test lc1[1][2] == vac
        @test lc1.basis == qpairbasis
        @test lc1.ħ == 2

        coh1 = coherentstate(qpairbasis, 1.0)
        coh2 = coherentstate(qpairbasis, -1.0)
        coeffs = [0.5, 0.5]
        states = [coh1, coh2]
        lc2 = GaussianLinearCombination(qpairbasis, coeffs, states)
        @test lc2 isa GaussianLinearCombination
        @test length(lc2) == 2
        @test lc2.coeffs == coeffs
        @test lc2.states == states

        pairs = [(0.6, coh1), (0.8, coh2)]
        lc3 = GaussianLinearCombination(pairs)
        @test lc3 isa GaussianLinearCombination
        @test length(lc3) == 2
        @test lc3.coeffs == [0.6, 0.8]

        lc4 = GaussianLinearCombination(0.6 => coh1, 0.8 => coh2)
        @test lc4 isa GaussianLinearCombination
        @test length(lc4) == 2
        @test lc4.coeffs == [0.6, 0.8]

        vac_block = vacuumstate(qblockbasis)
        lc_block = GaussianLinearCombination(vac_block)
        @test lc_block.basis == qblockbasis
        @test Gabs.nmodes(vac_block) == nmodes
    end
    
    @testset "Convenient GaussianState Arithmetic Interface" begin
        nmodes = rand(1:3)
        qpairbasis = QuadPairBasis(nmodes)
        qblockbasis = QuadBlockBasis(nmodes)
        
        @testset "Scalar multiplication of GaussianState" begin
            for basis in [qpairbasis, qblockbasis]
                state = coherentstate(basis, 1.0 + 0.5im)
                
                lc1 = 0.6 * state
                @test lc1 isa GaussianLinearCombination
                @test length(lc1) == 1
                @test lc1.basis == basis
                @test lc1.ħ == state.ħ
                @test lc1.coeffs[1] == 0.6
                @test lc1.states[1] == state
                
                lc2 = state * 0.6
                @test lc2 isa GaussianLinearCombination
                @test lc1 == lc2
                
                lc3 = (0.5 + 0.3im) * state
                @test lc3 isa GaussianLinearCombination
                @test lc3.coeffs[1] == 0.5 + 0.3im
                @test eltype(lc3.coeffs) <: Complex
                
                lc_explicit = GaussianLinearCombination(basis, [0.6], [state])
                @test isapprox(lc1, lc_explicit)
            end
        end
        
        @testset "Addition of GaussianStates" begin
            for basis in [qpairbasis, qblockbasis]
                state1 = coherentstate(basis, 1.0)
                state2 = coherentstate(basis, -1.0)
                
                lc = state1 + state2
                @test lc isa GaussianLinearCombination
                @test length(lc) == 2
                @test lc.basis == basis
                @test lc.ħ == state1.ħ
                @test lc.coeffs == [1.0, 1.0]
                @test lc.states == [state1, state2]
                
                lc_explicit = GaussianLinearCombination(basis, [1.0, 1.0], [state1, state2])
                @test lc == lc_explicit
                
                vac = vacuumstate(basis)
                sq = squeezedstate(basis, 0.3, π/4)
                lc2 = vac + sq
                @test lc2 isa GaussianLinearCombination
                @test length(lc2) == 2
                @test lc2.states == [vac, sq]
            end
        end
        
        @testset "Subtraction of GaussianStates" begin
            for basis in [qpairbasis, qblockbasis]
                state1 = coherentstate(basis, 1.0)
                state2 = coherentstate(basis, -1.0)
                
                lc = state1 - state2
                @test lc isa GaussianLinearCombination
                @test length(lc) == 2
                @test lc.basis == basis
                @test lc.coeffs == [1.0, -1.0]
                @test lc.states == [state1, state2]
                
                lc_explicit = GaussianLinearCombination(basis, [1.0, -1.0], [state1, state2])
                @test lc == lc_explicit
            end
        end
        
        @testset "Negation of GaussianState" begin
            for basis in [qpairbasis, qblockbasis]
                state = coherentstate(basis, 1.0 + 0.5im)
                
                lc = -state
                @test lc isa GaussianLinearCombination
                @test length(lc) == 1
                @test lc.coeffs[1] == -1.0
                @test lc.states[1] == state
                
                lc_equiv = (-1) * state
                @test lc == lc_equiv
            end
        end
        
        @testset "Mixed operations: GaussianState with GaussianLinearCombination" begin
            for basis in [qpairbasis, qblockbasis]
                state1 = coherentstate(basis, 1.0)
                state2 = coherentstate(basis, -1.0)
                state3 = vacuumstate(basis)
                
                lc = GaussianLinearCombination(basis, [0.6, 0.8], [state1, state2])
                
                lc_plus = state3 + lc
                @test lc_plus isa GaussianLinearCombination
                @test length(lc_plus) == 3
                @test lc_plus.coeffs == [1.0, 0.6, 0.8]
                @test lc_plus.states == [state3, state1, state2]
                
                lc_plus2 = lc + state3
                @test lc_plus2 isa GaussianLinearCombination
                @test length(lc_plus2) == 3
                @test lc_plus2.coeffs == [0.6, 0.8, 1.0]
                @test lc_plus2.states == [state1, state2, state3]
                
                lc_minus = state3 - lc
                @test lc_minus isa GaussianLinearCombination
                @test length(lc_minus) == 3
                @test lc_minus.coeffs == [1.0, -0.6, -0.8]
                @test lc_minus.states == [state3, state1, state2]
                
                lc_minus2 = lc - state3
                @test lc_minus2 isa GaussianLinearCombination
                @test length(lc_minus2) == 3
                @test lc_minus2.coeffs == [0.6, 0.8, -1.0]
                @test lc_minus2.states == [state1, state2, state3]
            end
        end
        
        @testset "Complex expressions and equivalences" begin
            basis = qpairbasis
            state1 = coherentstate(basis, 1.0)
            state2 = coherentstate(basis, -1.0)
            state3 = vacuumstate(basis)
            
            lc_complex = 0.5 * state1 + 0.3 * state2 - 0.2 * state3
            @test lc_complex isa GaussianLinearCombination
            @test length(lc_complex) == 3
            
            lc_step1 = 0.5 * state1
            lc_step2 = lc_step1 + 0.3 * state2
            lc_step3 = lc_step2 - 0.2 * state3
            @test isapprox(lc_complex.coeffs, lc_step3.coeffs)
            
            cat_state = 0.5 * state1 + 0.5 * state2
            @test cat_state isa GaussianLinearCombination
            @test length(cat_state) == 2
            @test cat_state.coeffs == [0.5, 0.5]
            @test cat_state.states == [state1, state2]
            
            old_approach = 0.5 * GaussianLinearCombination(state1) + 0.5 * GaussianLinearCombination(state2)
            @test isapprox(cat_state.coeffs, old_approach.coeffs)
        end
        
        @testset "Error handling for incompatible states" begin
            state_pair = coherentstate(qpairbasis, 1.0)
            state_block = coherentstate(qblockbasis, 1.0)
            state_h1 = coherentstate(qpairbasis, 1.0, ħ=1)
            state_h2 = coherentstate(qpairbasis, 1.0, ħ=2)
            
            @test_throws ArgumentError state_pair + state_block
            @test_throws ArgumentError state_pair - state_block
            
            @test_throws ArgumentError state_h1 + state_h2
            @test_throws ArgumentError state_h1 - state_h2
            
            lc_pair = GaussianLinearCombination(state_pair)
            lc_h1 = GaussianLinearCombination(state_h1)
            
            @test_throws ArgumentError state_block + lc_pair
            @test_throws ArgumentError state_h2 + lc_h1
            @test_throws ArgumentError state_pair - lc_h1
            @test_throws ArgumentError lc_pair - state_h1
        end
        
        @testset "Type promotion and coefficient types" begin
            basis = qpairbasis
            state = coherentstate(basis, 1.0)
            
            lc_int = 2 * state
            @test lc_int.coeffs[1] == 2.0
            @test eltype(lc_int.coeffs) <: Real
            
            lc_float = 2.5 * state
            @test lc_float.coeffs[1] == 2.5
            
            lc_complex = (1.0 + 2.0im) * state
            @test lc_complex.coeffs[1] == 1.0 + 2.0im
            @test eltype(lc_complex.coeffs) <: Complex
            
            lc_mixed = lc_int + lc_complex
            @test eltype(lc_mixed.coeffs) <: Complex
        end
        
        @testset "StaticArrays compatibility" begin
            coh_static = coherentstate(SVector{2*nmodes}, SMatrix{2*nmodes,2*nmodes}, qpairbasis, 1.0)
            vac_static = vacuumstate(SVector{2*nmodes}, SMatrix{2*nmodes,2*nmodes}, qpairbasis)
            
            lc_static_mult = 0.7 * coh_static
            @test lc_static_mult isa GaussianLinearCombination
            @test lc_static_mult.states[1].mean isa SVector
            @test lc_static_mult.states[1].covar isa SMatrix
            
            lc_static_add = coh_static + vac_static
            @test lc_static_add isa GaussianLinearCombination
            @test lc_static_add.states[1].mean isa SVector
            @test lc_static_add.states[2].covar isa SMatrix
            
            lc_regular = GaussianLinearCombination(coherentstate(qpairbasis, -1.0))
            lc_mixed = coh_static + lc_regular
            @test lc_mixed isa GaussianLinearCombination
            @test length(lc_mixed) == 2
        end
        
        @testset "Real-world usage examples" begin
            basis = QuadPairBasis(1)
            
            state_pos = coherentstate(basis, 1.0 + 1.0im)
            state_neg = coherentstate(basis, 1.0 - 1.0im)
            
            superposition = 0.5 * state_pos + 0.5 * state_neg
            @test superposition isa GaussianLinearCombination
            @test length(superposition) == 2
            
            vac = vacuumstate(basis)
            coh = coherentstate(basis, 2.0)
            sq = squeezedstate(basis, 0.5, π/4)
            
            complex_state = 0.6 * vac + 0.3 * coh - 0.1 * sq
            @test complex_state isa GaussianLinearCombination
            @test length(complex_state) == 3
            @test isapprox(complex_state.coeffs, [0.6, 0.3, -0.1])
            
            Gabs.normalize!(complex_state)
            norm_val = sqrt(sum(abs2, complex_state.coeffs))
            @test isapprox(norm_val, 1.0, atol=1e-15)
        end
    end

    @testset "Constructor validation" begin
        vac = vacuumstate(qpairbasis)
        vac_block = vacuumstate(qblockbasis)
        vac_diff_h = vacuumstate(qpairbasis, ħ = 1)

        @test_throws ArgumentError GaussianLinearCombination(qpairbasis, [1.0, 1.0], [vac, vac_block])
        
        @test_throws ArgumentError GaussianLinearCombination(qpairbasis, [1.0, 1.0], [vac, vac_diff_h])
        
        @test_throws DimensionMismatch GaussianLinearCombination(qpairbasis, [1.0], [vac, vac])
        
        @test_throws ArgumentError GaussianLinearCombination(qpairbasis, Float64[], GaussianState[])
        @test_throws ArgumentError GaussianLinearCombination(Tuple{Float64,GaussianState}[])
    end

    @testset "Arithmetic operations" begin
        
        coh1 = coherentstate(qpairbasis, 1.0)
        coh2 = coherentstate(qpairbasis, -1.0)
        lc1 = GaussianLinearCombination(coh1)
        lc2 = GaussianLinearCombination(coh2)

        lc_sum = lc1 + lc2
        @test lc_sum isa GaussianLinearCombination
        @test length(lc_sum) == 2
        @test lc_sum.coeffs == [1.0, 1.0]
        @test lc_sum.states == [coh1, coh2]

        lc_scaled1 = 0.5 * lc1
        @test lc_scaled1 isa GaussianLinearCombination
        @test lc_scaled1.coeffs == [0.5]
        @test lc_scaled1.states == [coh1]

        lc_scaled2 = lc1 * 0.5
        @test lc_scaled2 isa GaussianLinearCombination
        @test lc_scaled2.coeffs == [0.5]
        @test lc_scaled2 == lc_scaled1

        lc_complex = (1.0 + 2.0im) * lc1
        @test lc_complex.coeffs == [1.0 + 2.0im]

        lc_diff = lc1 - lc2
        @test lc_diff isa GaussianLinearCombination
        @test length(lc_diff) == 2
        @test lc_diff.coeffs == [1.0, -1.0]

        lc_neg = -lc1
        @test lc_neg isa GaussianLinearCombination
        @test lc_neg.coeffs == [-1.0]
    end

    @testset "Arithmetic validation" begin
        vac = vacuumstate(qpairbasis)
        vac_block = vacuumstate(qblockbasis)
        vac_diff_h = vacuumstate(qpairbasis, ħ = 1)
        
        lc1 = GaussianLinearCombination(vac)
        lc_block = GaussianLinearCombination(vac_block)
        lc_diff_h = GaussianLinearCombination(vac_diff_h)

        @test_throws ArgumentError lc1 + lc_block
        @test_throws ArgumentError lc1 - lc_block
        
        @test_throws ArgumentError lc1 + lc_diff_h
        @test_throws ArgumentError lc1 - lc_diff_h
    end

    @testset "Utility functions" begin
        coh1 = coherentstate(qpairbasis, 1.0)
        coh2 = coherentstate(qpairbasis, -1.0)
        lc = GaussianLinearCombination(qpairbasis, [0.6, 0.8], [coh1, coh2])

        @test length(lc) == 2

        @test lc[1] == (0.6, coh1)
        @test lc[2] == (0.8, coh2)

        pairs = collect(lc)
        @test pairs == [(0.6, coh1), (0.8, coh2)]
        
        coeffs_iter = [c for (c, s) in lc]
        states_iter = [s for (c, s) in lc]
        @test coeffs_iter == [0.6, 0.8]
        @test states_iter == [coh1, coh2]
    end

    @testset "Normalization" begin
        coh1 = coherentstate(qpairbasis, 1.0)
        coh2 = coherentstate(qpairbasis, -1.0)
        lc = GaussianLinearCombination(qpairbasis, [3.0, 4.0], [coh1, coh2])
        
        initial_norm = sqrt(sum(abs2, lc.coeffs))
        @test initial_norm == 5.0
        
        result = Gabs.normalize!(lc)
        @test result === lc  
        @test isapprox(sqrt(sum(abs2, lc.coeffs)), 1.0, atol=1e-15)
        @test lc.coeffs ≈ [0.6, 0.8]

        lc_zero = GaussianLinearCombination(qpairbasis, [0.0, 0.0], [coh1, coh2])
        Gabs.normalize!(lc_zero)
        @test all(lc_zero.coeffs .== 0.0)
    end

    @testset "Simplification" begin
        coh1 = coherentstate(qpairbasis, 1.0)
        coh2 = coherentstate(qpairbasis, -1.0)
        vac = vacuumstate(qpairbasis)

        lc1 = GaussianLinearCombination(qpairbasis, [1.0, 1e-16, 0.5], [coh1, coh2, vac])
        Gabs.simplify!(lc1)
        @test length(lc1) == 2
        @test lc1.coeffs ≈ [1.0, 0.5]
        @test lc1.states == [coh1, vac]

        lc2 = GaussianLinearCombination(qpairbasis, [0.5, 0.3, 0.2], [coh1, coh2, coh1])
        Gabs.simplify!(lc2)
        @test length(lc2) == 2
        if lc2.states[1] == coh1
            @test lc2.coeffs ≈ [0.7, 0.3]
            @test lc2.states == [coh1, coh2]
        else
            @test lc2.coeffs ≈ [0.3, 0.7]
            @test lc2.states == [coh2, coh1]
        end

        lc3 = GaussianLinearCombination(qpairbasis, [1e-16, -1e-16], [coh1, coh2])
        Gabs.simplify!(lc3)
        @test length(lc3) == 1
        @test lc3.coeffs[1] ≈ 1e-14
        @test lc3.states[1] == vacuumstate(qpairbasis)

        lc4 = GaussianLinearCombination(qpairbasis, [1.0, -1.0], [coh1, coh1])
        Gabs.simplify!(lc4)
        @test length(lc4) == 1
        @test lc4.coeffs[1] ≈ 1e-14

        lc5 = GaussianLinearCombination(qpairbasis, [1.0, 1e-10], [coh1, coh2])
        Gabs.simplify!(lc5, atol=1e-12)
        @test length(lc5) == 2  # 1e-10 > 1e-12, so keep both
        Gabs.simplify!(lc5, atol=1e-8)
        @test length(lc5) == 1   # 1e-10 < 1e-8, so remove small one
    end

    @testset "Equality and approximation" begin
        coh1 = coherentstate(qpairbasis, 1.0)
        coh2 = coherentstate(qpairbasis, -1.0)
        
        lc1 = GaussianLinearCombination(qpairbasis, [0.6, 0.8], [coh1, coh2])
        lc2 = GaussianLinearCombination(qpairbasis, [0.6, 0.8], [coh1, coh2])
        lc3 = GaussianLinearCombination(qpairbasis, [0.6, 0.8001], [coh1, coh2])
        
        @test lc1 == lc2
        @test lc1 != lc3
        
        @test isapprox(lc1, lc3, atol=1e-3)
        @test !isapprox(lc1, lc3, atol=1e-5)

        lc_block = GaussianLinearCombination(qblockbasis, [0.6, 0.8], 
                                           [coherentstate(qblockbasis, 1.0), coherentstate(qblockbasis, -1.0)])
        @test lc1 != lc_block
        @test !isapprox(lc1, lc_block)
    end

    @testset "Display methods" begin
        coh1 = coherentstate(qpairbasis, 1.0)
        coh2 = coherentstate(qpairbasis, -1.0)
        lc = GaussianLinearCombination(qpairbasis, [0.6, 0.8], [coh1, coh2])
        
        io = IOBuffer()
        show(io, lc)
        output_short = String(take!(io))
        @test contains(output_short, "GaussianLinearCombination")
        
        show(io, MIME("text/plain"), lc)
        output_long = String(take!(io))
        @test contains(output_long, "GaussianLinearCombination")
        @test contains(output_long, "QuadPairBasis")
        @test contains(output_long, "mode")
    end

    @testset "Complex coefficients" begin
        coh = coherentstate(qpairbasis, 1.0)
        vac = vacuumstate(qpairbasis)
        
        lc_complex = GaussianLinearCombination(qpairbasis, [1.0 + 2.0im, 0.5 - 1.0im], [coh, vac])
        @test lc_complex isa GaussianLinearCombination
        @test eltype(lc_complex.coeffs) <: Complex
        
        lc_scaled = (0.5 + 0.5im) * lc_complex
        @test eltype(lc_scaled.coeffs) <: Complex
        
        Gabs.normalize!(lc_complex)
        norm_val = sqrt(sum(abs2, lc_complex.coeffs))
        @test isapprox(norm_val, 1.0, atol=1e-15)
    end

    @testset "Integration with existing Gabs types" begin
        for basis in [qpairbasis, qblockbasis]
            coh = coherentstate(basis, 2.0)
            vac = vacuumstate(basis)
            sq = squeezedstate(basis, 0.5, π/4)
            
            lc = GaussianLinearCombination(basis, [0.5, 0.3, 0.2], [coh, vac, sq])
            @test lc.basis == basis
            @test all(s.basis == basis for s in lc.states)
        end
        
        coh_h1 = coherentstate(qpairbasis, 1.0, ħ = 1)
        coh_h2 = coherentstate(qpairbasis, -1.0, ħ = 1)
        lc_h1 = GaussianLinearCombination(qpairbasis, [0.6, 0.8], [coh_h1, coh_h2])
        @test lc_h1.ħ == 1
        
        lc_h1_scaled = 2.0 * lc_h1
        @test lc_h1_scaled.ħ == 1
    end

    @testset "Static arrays compatibility" begin
        coh_static = coherentstate(SVector{2*nmodes}, SMatrix{2*nmodes,2*nmodes}, qpairbasis, 1.0)
        vac_static = vacuumstate(SVector{2*nmodes}, SMatrix{2*nmodes,2*nmodes}, qpairbasis)
        
        lc_static = GaussianLinearCombination(qpairbasis, [0.6, 0.8], [coh_static, vac_static])
        @test lc_static isa GaussianLinearCombination
        @test lc_static.states[1].mean isa SVector
        @test lc_static.states[1].covar isa SMatrix
        
        lc_static_scaled = 0.5 * lc_static
        @test lc_static_scaled isa GaussianLinearCombination
    end

    @testset "Cat state example" begin
        basis = QuadPairBasis(1)
        state1 = coherentstate(basis, 1.0)
        state2 = coherentstate(basis, -1.0)
        
        lcgs = 0.5 * GaussianLinearCombination(state1) + 0.5 * GaussianLinearCombination(state2)
        
        @test lcgs isa GaussianLinearCombination
        @test length(lcgs) == 2
        @test lcgs.coeffs == [0.5, 0.5]
        @test lcgs.states[1] == state1
        @test lcgs.states[2] == state2
        
        norm_val = sqrt(sum(abs2, lcgs.coeffs))
        @test isapprox(norm_val, sqrt(0.5), atol=1e-15)
        
        cat_state = GaussianLinearCombination(0.5 => state1, 0.5 => state2)
        @test cat_state.coeffs == [0.5, 0.5]
    end

    @testset "Edge cases" begin
        coh1 = coherentstate(qpairbasis, 1.0)
        coh2 = coherentstate(qpairbasis, -1.0)
        
        lc_single = GaussianLinearCombination(coh1)
        @test length(lc_single) == 1
        @test lc_single.coeffs == [1.0]
        
        lc_with_zero = GaussianLinearCombination(qpairbasis, [0.0, 1.0], [coh1, coh2])
        @test length(lc_with_zero) == 2
        Gabs.simplify!(lc_with_zero)
        @test length(lc_with_zero) == 1
        @test lc_with_zero.states[1] == coh2
        
        many_states = [coherentstate(qpairbasis, Float64(i)) for i in 1:100]
        many_coeffs = [1.0/100 for _ in 1:100]
        lc_many = GaussianLinearCombination(qpairbasis, many_coeffs, many_states)
        @test length(lc_many) == 100
        
        lc_identical = GaussianLinearCombination(qpairbasis, [0.2, 0.3, 0.5], [coh1, coh1, coh1])
        Gabs.simplify!(lc_identical)
        @test length(lc_identical) == 1
        @test lc_identical.coeffs[1] ≈ 1.0
        @test lc_identical.states[1] == coh1
    end
end