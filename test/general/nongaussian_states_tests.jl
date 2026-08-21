@testitem "Non-Gaussian States" begin
    using Gabs
    using LinearAlgebra

    @testset "Cat States - Basic Properties" begin
        basis = QuadPairBasis(1)
        α = 1.0 + 0.5im
        cat_even = catstate_even(basis, α)
        @test cat_even isa GaussianLinearCombination
        @test length(cat_even) == 2
        @test cat_even.basis == basis
        @test cat_even.ħ == 2
        c1, c2 = cat_even.coeffs
        @test abs(c1 - c2) < 1e-12
        @test c1 > 0  
        cat_odd = catstate_odd(basis, α)
        @test cat_odd isa GaussianLinearCombination
        @test length(cat_odd) == 2
        c1, c2 = cat_odd.coeffs
        @test abs(c1 + c2) < 1e-12 
        @test abs(c1) ≈ abs(c2)     
        phase = π/3
        cat_general = catstate(basis, α, phase)
        @test cat_general isa GaussianLinearCombination
        @test length(cat_general) == 2
        cat_even_check = catstate(basis, α, 0.0)
        @test isapprox(cat_even.coeffs, cat_even_check.coeffs, atol=1e-12)
        cat_odd_check = catstate(basis, α, π)
        @test isapprox(cat_odd.coeffs[1], cat_odd_check.coeffs[1], atol=1e-12)
        @test isapprox(cat_odd.coeffs[2], cat_odd_check.coeffs[2], atol=1e-12)
    end
    
    @testset "Cat States - Squeezed Variants" begin
        basis = QuadPairBasis(1)
        α = 1.0
        r, θ = 0.5, π/4
        cat_squeezed_even = catstate_even(basis, α, squeeze_params=(r, θ))
        @test cat_squeezed_even isa GaussianLinearCombination
        @test length(cat_squeezed_even) == 2
        cat_squeezed_odd = catstate_odd(basis, α, squeeze_params=(r, θ))
        @test cat_squeezed_odd isa GaussianLinearCombination
        @test length(cat_squeezed_odd) == 2
        phase = π/6
        cat_squeezed_general = catstate(basis, α, phase, squeeze_params=(r, θ))
        @test cat_squeezed_general isa GaussianLinearCombination
        @test length(cat_squeezed_general) == 2
        regular_cat = catstate_even(basis, α)
        @test !isapprox(cat_squeezed_even.states[1].covar, regular_cat.states[1].covar)
    end
    
    @testset "Cat States - Multi-mode" begin
        basis = QuadPairBasis(2)
        αs = [1.0 + 0.5im, 0.8 - 0.3im]
        cat_multi_even = catstate_even(basis, αs)
        @test cat_multi_even isa GaussianLinearCombination
        @test length(cat_multi_even) == 4  
        @test cat_multi_even.basis == basis
        cat_multi_odd = catstate_odd(basis, αs)
        @test cat_multi_odd isa GaussianLinearCombination
        @test length(cat_multi_odd) == 4
        phases = [π/4, π/6]
        cat_multi_general = catstate(basis, αs, phases)
        @test cat_multi_general isa GaussianLinearCombination
        @test length(cat_multi_general) == 4
        squeeze_params = [(0.3, π/8), (0.4, π/3)]
        cat_multi_squeezed = catstate_even(basis, αs, squeeze_params=squeeze_params)
        @test cat_multi_squeezed isa GaussianLinearCombination
        @test length(cat_multi_squeezed) == 4
    end
    
    @testset "Cat States - Normalization" begin
        basis = QuadPairBasis(1)
        for α in [0.5, 1.0, 1.5, 2.0]
            cat_even = catstate_even(basis, α)
            cat_odd = catstate_odd(basis, α)
            norm_even = sqrt(sum(abs2, cat_even.coeffs))
            norm_odd = sqrt(sum(abs2, cat_odd.coeffs))
            if α > 1.0
                @test abs(cat_even.coeffs[1] - cat_even.coeffs[2]) < 1e-10
                @test abs(abs(cat_odd.coeffs[1]) - abs(cat_odd.coeffs[2])) < 1e-10
            end
        end
    end
    
    @testset "Cat States - Edge Cases" begin
        basis = QuadPairBasis(1)
        α_small = 0.1
        cat_even_small = catstate_even(basis, α_small)
        cat_odd_small = catstate_odd(basis, α_small)
        @test cat_even_small isa GaussianLinearCombination
        @test cat_odd_small isa GaussianLinearCombination
        @test abs(cat_even_small.coeffs[1]) < abs(cat_odd_small.coeffs[1])
        α_large = 3.0
        cat_even_large = catstate_even(basis, α_large)
        cat_odd_large = catstate_odd(basis, α_large)
        @test cat_even_large isa GaussianLinearCombination
        @test cat_odd_large isa GaussianLinearCombination
        expected_coeff = 1/sqrt(2)
        @test abs(abs(cat_even_large.coeffs[1]) - expected_coeff) < 0.01
        @test abs(abs(cat_odd_large.coeffs[1]) - expected_coeff) < 0.01
        α_imag = 1.5im
        cat_imag = catstate_even(basis, α_imag)
        @test cat_imag isa GaussianLinearCombination
        @test length(cat_imag) == 2
    end
    
    @testset "GKP States - Basic Properties" begin
        basis = QuadPairBasis(1)
        gkp_square = gkpstate(basis, lattice="square", delta=0.2, nmax=3)
        @test gkp_square isa GaussianLinearCombination
        @test length(gkp_square) == 7 
        @test gkp_square.basis == basis
        @test gkp_square.ħ == 2
        gkp_hex = gkpstate(basis, lattice="hexagonal", delta=0.2, nmax=2)
        @test gkp_hex isa GaussianLinearCombination
        @test length(gkp_hex) > 0 
        @test gkp_hex.basis == basis
        gkp_small_delta = gkpstate(basis, lattice="square", delta=0.1, nmax=2)
        gkp_large_delta = gkpstate(basis, lattice="square", delta=0.3, nmax=2)
        @test gkp_small_delta isa GaussianLinearCombination
        @test gkp_large_delta isa GaussianLinearCombination
        @test length(gkp_small_delta) == length(gkp_large_delta)  
        gkp_small_nmax = gkpstate(basis, lattice="square", delta=0.2, nmax=2)
        gkp_large_nmax = gkpstate(basis, lattice="square", delta=0.2, nmax=4)
        @test length(gkp_small_nmax) < length(gkp_large_nmax)  
    end
    
    @testset "GKP States - Different Lattices" begin
        basis = QuadPairBasis(1)
        delta = 0.15
        nmax = 2
        gkp_square = gkpstate(basis, lattice="square", delta=delta, nmax=nmax)
        gkp_hex = gkpstate(basis, lattice="hexagonal", delta=delta, nmax=nmax)
        @test gkp_square isa GaussianLinearCombination
        @test gkp_hex isa GaussianLinearCombination
        @test_throws ArgumentError gkpstate(basis, lattice=:triangular)
    end
    
    @testset "GKP States - Custom Parameters" begin
        basis = QuadPairBasis(1)
        
        gkp_custom_hbar = gkpstate(basis, lattice="square", delta=0.2, nmax=2, ħ=1)
        @test gkp_custom_hbar.ħ == 1
        
        basis_block = QuadBlockBasis(1)
        gkp_block = gkpstate(basis_block, lattice="square", delta=0.2, nmax=2)
        @test gkp_block.basis == basis_block
    end
    
    @testset "Helper Functions" begin
        basis = QuadPairBasis(1)
        
        state1 = coherentstate(basis, 1.0)
        state2 = coherentstate(basis, -1.0)
        states = [state1, state2]
        coeffs = [1.0, 1.0]
        
        norm_val = norm_factor(states, coeffs)
        @test norm_val isa Float64
        @test norm_val > 0
        
        coeffs_weighted = [0.6, 0.8]
        norm_factor_weighted = norm_factor(states, coeffs_weighted)
        @test norm_factor_weighted isa Float64
        @test norm_factor_weighted > 0
        
        gkp1 = gkpstate(basis, lattice="square", delta=0.1, nmax=3)
        gkp2 = gkpstate(basis, lattice="square", delta=0.2, nmax=2)
    end
    
    @testset "Integration with Existing Framework" begin
        basis = QuadPairBasis(1)
        cat = catstate_even(basis, 1.0)
        Gabs.simplify!(cat)
        @test cat isa GaussianLinearCombination
        cat_copy = catstate_odd(basis, 1.5)
        original_norm = sqrt(sum(abs2, cat_copy.coeffs))
        Gabs.normalize!(cat_copy)
        new_norm = sqrt(sum(abs2, cat_copy.coeffs))
        @test abs(new_norm - 1.0) < 1e-12
        cat1 = catstate_even(basis, 1.0)
        cat2 = catstate_odd(basis, 1.0)
        cat_sum = cat1 + cat2
        @test cat_sum isa GaussianLinearCombination
        @test length(cat_sum) <= 4 
        cat_scaled = 0.5 * cat1
        @test cat_scaled isa GaussianLinearCombination
        @test all(abs.(cat_scaled.coeffs) .≈ 0.5 * abs.(cat1.coeffs))
        @test cat1 == cat1
        @test isapprox(cat1, cat1)
    end
    
    @testset "Multi-mode Tensor Products" begin
        basis1 = QuadPairBasis(1)
        basis2 = QuadPairBasis(2)
        
        α1, α2 = 1.0, 0.8im
        cat1 = catstate_even(basis1, α1)
        cat2 = catstate_even(basis1, α2)
        
        cat_tensor_manual = Gabs._tensor(cat1, cat2)
        @test cat_tensor_manual isa GaussianLinearCombination
        @test length(cat_tensor_manual) == 4  # 2 × 2 = 4
        @test cat_tensor_manual.basis == basis2
        
        cat_multi = catstate_even(basis2, [α1, α2])
        @test cat_multi isa GaussianLinearCombination
        @test length(cat_multi) == 4
        @test cat_multi.basis == basis2
        
        @test length(cat_tensor_manual) == length(cat_multi)
    end
    
    @testset "Numerical Stability" begin
        basis = QuadPairBasis(1)
        
        cat_tiny = catstate_even(basis, 1e-8)
        @test cat_tiny isa GaussianLinearCombination
        @test all(isfinite.(cat_tiny.coeffs))
        
        cat_large = catstate_even(basis, 10.0)
        @test cat_large isa GaussianLinearCombination
        @test all(isfinite.(cat_large.coeffs))
        
        gkp_small_delta = gkpstate(basis, lattice="square", delta=1e-6, nmax=1)
        @test gkp_small_delta isa GaussianLinearCombination
        @test all(isfinite.(gkp_small_delta.coeffs))
        
        gkp_large_delta = gkpstate(basis, lattice="square", delta=2.0, nmax=1)
        @test gkp_large_delta isa GaussianLinearCombination
        @test all(isfinite.(gkp_large_delta.coeffs))
    end
    
    @testset "Error Handling" begin
        basis = QuadPairBasis(2)
        
        αs_wrong = [1.0]  
        @test_throws AssertionError catstate_even(basis, αs_wrong)
        
        phases_wrong = [π/4] 
        @test_throws AssertionError catstate(basis, [1.0, 1.0], phases_wrong)
        
        squeeze_params_wrong = [(0.5, π/4)]  
        @test_throws AssertionError catstate_even(basis, [1.0, 1.0], squeeze_params=squeeze_params_wrong)
        
        @test_throws ArgumentError gkpstate(basis, lattice=:invalid)
        
        state1 = coherentstate(QuadPairBasis(1), 1.0)
        states = [state1, state1]
        coeffs_wrong = [1.0]  
        @test_throws AssertionError norm_factor(states, coeffs_wrong)
    end

    @testset "GKP state validation coverage" begin
        basis = QuadPairBasis(1)
        
        @test_throws ArgumentError gkpstate(basis, lattice="square", delta=0.0, nmax=2)
        @test_throws ArgumentError gkpstate(basis, lattice="square", delta=-0.1, nmax=2)
        
        gkp_tiny_delta = gkpstate(basis, lattice="square", delta=1e-8, nmax=1)
        @test gkp_tiny_delta isa GaussianLinearCombination
        
        @test_throws ArgumentError gkpstate(basis, lattice="square", delta=0.1, nmax=0)
        @test_throws ArgumentError gkpstate(basis, lattice="square", delta=0.1, nmax=-1)
        
        gkp_large_nmax = gkpstate(basis, lattice="square", delta=0.3, nmax=55)
        @test gkp_large_nmax isa GaussianLinearCombination
        @test length(gkp_large_nmax) == 111
    end

    @testset "Multi-mode catstate_odd squeeze params validation coverage" begin
        basis = QuadPairBasis(2)
        αs = [1.0 + 0.5im, 0.8 - 0.3im]
        
        correct_squeeze_params = [(0.3, π/8), (0.4, π/3)]
        cat_with_squeeze = catstate_odd(basis, αs, squeeze_params=correct_squeeze_params)
        @test cat_with_squeeze isa GaussianLinearCombination
        @test length(cat_with_squeeze) == 4  
        
        wrong_squeeze_params = [(0.3, π/8)] 
        
        @test_throws AssertionError catstate_odd(basis, αs, squeeze_params=wrong_squeeze_params)
        
        cat_no_squeeze = catstate_odd(basis, αs)
        @test cat_no_squeeze isa GaussianLinearCombination
        @test length(cat_no_squeeze) == 4
        
        basis3 = QuadPairBasis(3)
        αs3 = [1.0, 0.5im, -0.8]
        wrong_squeeze_params3 = [(0.3, π/8), (0.4, π/3)]  
        
        @test_throws AssertionError catstate_odd(basis3, αs3, squeeze_params=wrong_squeeze_params3)
    end

    @testset "Multi-mode catstate general squeeze params validation coverage" begin
        basis = QuadPairBasis(2)
        αs = [1.0 + 0.5im, 0.8 - 0.3im]
        phases = [π/4, π/6]
        
        correct_squeeze_params = [(0.3, π/8), (0.4, π/3)]
        cat_with_squeeze = catstate(basis, αs, phases, squeeze_params=correct_squeeze_params)
        @test cat_with_squeeze isa GaussianLinearCombination
        @test length(cat_with_squeeze) == 4  
        
        wrong_squeeze_params = [(0.3, π/8)]  
        
        @test_throws AssertionError catstate(basis, αs, phases, squeeze_params=wrong_squeeze_params)
        
        cat_no_squeeze = catstate(basis, αs, phases)
        @test cat_no_squeeze isa GaussianLinearCombination
        @test length(cat_no_squeeze) == 4
        
        basis1 = QuadPairBasis(1)
        αs1 = [1.0]
        phases1 = [π/3]
        wrong_squeeze_params1 = [(0.3, π/8), (0.4, π/3)] 
        
        @test_throws AssertionError catstate(basis1, αs1, phases1, squeeze_params=wrong_squeeze_params1)
        
        cat_empty_squeeze = catstate(basis, αs, phases, squeeze_params=nothing)
        @test cat_empty_squeeze isa GaussianLinearCombination
    end

    @testset "Additional edge cases for robustness" begin
        basis = QuadPairBasis(1)
        
        gkp_boundary = gkpstate(basis, lattice="square", delta=1e-6, nmax=50)
        @test gkp_boundary isa GaussianLinearCombination
        
        state1 = coherentstate(basis, 1.0)
        state2 = coherentstate(basis, -1.0)
        
        small_coeffs = [1e-10, 1e-10]
        norm_small = norm_factor([state1, state2], small_coeffs)
        @test norm_small isa Float64
        @test norm_small > 0.0
        
        overlap_identical = Gabs._overlap(state1, state1)
        @test overlap_identical ≈ 1.0 + 0.0im
        
        cat_single = catstate_odd(QuadPairBasis(1), [1.0], squeeze_params=[(0.3, π/4)])
        @test cat_single isa GaussianLinearCombination
    end
    
end