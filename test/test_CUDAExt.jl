@testitem "GPU Foundation - State Creation" begin
    using CUDA
    using Gabs
    using Gabs: device
    using LinearAlgebra
    using Statistics

    @testset "GPU Vacuum State" begin
        basis = QuadPairBasis(1)
        vac_cpu = vacuumstate(basis)
        vac_gpu = vacuumstate(basis) |> gpu
        @test vac_gpu.mean isa CuVector{Float32}
        @test vac_gpu.covar isa CuMatrix{Float32}
        @test vac_gpu.basis == basis
        @test vac_gpu.ħ == 2
        @test device(vac_gpu) == :gpu
        @test device(vac_cpu) == :cpu
        @test Array(vac_gpu.mean) ≈ vac_cpu.mean
        @test Array(vac_gpu.covar) ≈ vac_cpu.covar
        vac_gpu_f64 = gpu(vacuumstate(basis), precision=Float64)
        @test vac_gpu_f64.mean isa CuVector{Float64}
        @test vac_gpu_f64.covar isa CuMatrix{Float64}
        vac_back_to_cpu = vac_gpu |> cpu
        @test device(vac_back_to_cpu) == :cpu
        @test vac_back_to_cpu.mean ≈ vac_cpu.mean
        @test vac_back_to_cpu.covar ≈ vac_cpu.covar
    end
    
    @testset "GPU Coherent State" begin
        basis = QuadPairBasis(1)
        α = 1.0f0 + 0.5f0im
        coh_cpu = coherentstate(basis, α)
        coh_gpu = coherentstate(basis, α) |> gpu
        @test coh_gpu.mean isa CuVector{Float32}
        @test coh_gpu.covar isa CuMatrix{Float32}
        @test device(coh_gpu) == :gpu
        @test Array(coh_gpu.mean) ≈ Float32.(coh_cpu.mean)
        @test Array(coh_gpu.covar) ≈ Float32.(coh_cpu.covar)
        α_gpu = CuArray([α])
        coh_auto = coherentstate(basis, α_gpu)
        @test device(coh_auto) == :gpu
        @test Array(coh_auto.mean) ≈ Float32.(coh_cpu.mean)
        basis_multi = QuadPairBasis(2)
        α_multi = [1.0f0 + 0.5f0im, -0.3f0 + 0.8f0im]
        coh_cpu_multi = coherentstate(basis_multi, α_multi)
        coh_gpu_multi = coherentstate(basis_multi, α_multi) |> gpu
        @test Array(coh_gpu_multi.mean) ≈ Float32.(coh_cpu_multi.mean)
        @test Array(coh_gpu_multi.covar) ≈ Float32.(coh_cpu_multi.covar)
        α_multi_gpu = CuArray(α_multi)
        coh_multi_auto = coherentstate(basis_multi, α_multi_gpu)
        @test device(coh_multi_auto) == :gpu
        @test Array(coh_multi_auto.mean) ≈ Float32.(coh_cpu_multi.mean)
    end
    
    @testset "GPU Squeezed State" begin
        basis = QuadPairBasis(1)
        r, θ = 0.3f0, Float32(π/4)
        sq_cpu = squeezedstate(basis, r, θ)
        sq_gpu = squeezedstate(basis, r, θ) |> gpu
        @test sq_gpu.mean isa CuVector{Float32}
        @test sq_gpu.covar isa CuMatrix{Float32}
        @test device(sq_gpu) == :gpu
        @test Array(sq_gpu.mean) ≈ Float32.(sq_cpu.mean)
        @test Array(sq_gpu.covar) ≈ Float32.(sq_cpu.covar) rtol=1e-6
        r_gpu = CuArray([r])
        sq_auto = squeezedstate(basis, r_gpu, θ)
        @test device(sq_auto) == :gpu
        @test Array(sq_auto.mean) ≈ Float32.(sq_cpu.mean)
        basis_multi = QuadPairBasis(2)
        r_vec = [0.3f0, 0.5f0]
        θ_vec = [Float32(π/4), Float32(π/6)]
        sq_cpu_multi = squeezedstate(basis_multi, r_vec, θ_vec)
        sq_gpu_multi = squeezedstate(basis_multi, r_vec, θ_vec) |> gpu
        @test Array(sq_gpu_multi.mean) ≈ Float32.(sq_cpu_multi.mean)
        @test Array(sq_gpu_multi.covar) ≈ Float32.(sq_cpu_multi.covar) rtol=1e-6
    end
    
    @testset "GPU Thermal State" begin
        basis = QuadPairBasis(1)
        n = 2.0f0
        thermal_cpu = thermalstate(basis, n)
        thermal_gpu = thermalstate(basis, n) |> gpu
        @test thermal_gpu.mean isa CuVector{Float32}
        @test thermal_gpu.covar isa CuMatrix{Float32}
        @test device(thermal_gpu) == :gpu
        @test Array(thermal_gpu.mean) ≈ Float32.(thermal_cpu.mean)
        @test Array(thermal_gpu.covar) ≈ Float32.(thermal_cpu.covar)
        n_gpu = CuArray([n])
        thermal_auto = thermalstate(basis, n_gpu)
        @test device(thermal_auto) == :gpu
        @test Array(thermal_auto.mean) ≈ Float32.(thermal_cpu.mean)
        basis_multi = QuadPairBasis(2)
        n_vec = [2.0f0, 3.0f0]
        thermal_cpu_multi = thermalstate(basis_multi, n_vec)
        thermal_gpu_multi = thermalstate(basis_multi, n_vec) |> gpu
        @test Array(thermal_gpu_multi.mean) ≈ Float32.(thermal_cpu_multi.mean)
        @test Array(thermal_gpu_multi.covar) ≈ Float32.(thermal_cpu_multi.covar)
    end
    
    @testset "GPU EPR State" begin
        basis = QuadPairBasis(2)
        r, θ = 0.5f0, Float32(π/3)
        epr_cpu = eprstate(basis, r, θ)
        epr_gpu = eprstate(basis, r, θ) |> gpu
        @test epr_gpu.mean isa CuVector{Float32}
        @test epr_gpu.covar isa CuMatrix{Float32}
        @test device(epr_gpu) == :gpu
        @test Array(epr_gpu.mean) ≈ Float32.(epr_cpu.mean)
        @test Array(epr_gpu.covar) ≈ Float32.(epr_cpu.covar) rtol=1e-6
    end
end

@testitem "GPU Foundation - Unitary Operations" begin
    using Gabs
    using Gabs: device
    using LinearAlgebra
    using CUDA
        
    @testset "GPU Displacement" begin
        basis = QuadPairBasis(1)
        α = 0.5f0 + 0.3f0im
        disp_cpu = displace(basis, α)
        disp_gpu = displace(basis, α) |> gpu
        @test disp_gpu.disp isa CuVector{Float32}
        @test disp_gpu.symplectic isa CuMatrix{Float32}
        @test device(disp_gpu) == :gpu
        @test Array(disp_gpu.disp) ≈ Float32.(disp_cpu.disp)
        @test Array(disp_gpu.symplectic) ≈ Float32.(disp_cpu.symplectic)
        α_gpu = CuArray([α])
        disp_auto = displace(basis, α_gpu)
        @test device(disp_auto) == :gpu
        @test Array(disp_auto.disp) ≈ Float32.(disp_cpu.disp)
        vac_gpu = vacuumstate(basis) |> gpu
        displaced_gpu = disp_gpu * vac_gpu
        vac_cpu = vacuumstate(basis)
        displaced_cpu = disp_cpu * vac_cpu
        @test Array(displaced_gpu.mean) ≈ Float32.(displaced_cpu.mean) rtol=1e-6
        @test Array(displaced_gpu.covar) ≈ Float32.(displaced_cpu.covar) rtol=1e-6
    end
    
    @testset "Mixed Device Operations" begin
        basis = QuadPairBasis(1)
        vac_cpu = vacuumstate(basis)
        disp_gpu = displace(basis, 1.0f0) |> gpu
        result = disp_gpu * vac_cpu
        @test device(result) == :gpu
        vac_gpu = vacuumstate(basis) |> gpu
        disp_cpu = displace(basis, 1.0f0)
        result2 = disp_cpu * vac_gpu
        @test device(result2) == :gpu
    end
    
    @testset "GPU Squeeze" begin
        basis = QuadPairBasis(1)
        r, θ = 0.3f0, Float32(π/4)
        squeeze_cpu = squeeze(basis, r, θ)
        squeeze_gpu = squeeze(basis, r, θ) |> gpu
        @test Array(squeeze_gpu.disp) ≈ Float32.(squeeze_cpu.disp)
        @test Array(squeeze_gpu.symplectic) ≈ Float32.(squeeze_cpu.symplectic) rtol=1e-6
        @test device(squeeze_gpu) == :gpu
        r_gpu = CuArray([r])
        squeeze_auto = squeeze(basis, r_gpu, θ)
        @test device(squeeze_auto) == :gpu
        vac_gpu = vacuumstate(basis) |> gpu
        squeezed_gpu = squeeze_gpu * vac_gpu
        vac_cpu = vacuumstate(basis)
        squeezed_cpu = squeeze_cpu * vac_cpu
        @test Array(squeezed_gpu.mean) ≈ Float32.(squeezed_cpu.mean) rtol=1e-6
        @test Array(squeezed_gpu.covar) ≈ Float32.(squeezed_cpu.covar) rtol=1e-5
    end
    
    @testset "GPU Phase Shift" begin
        basis = QuadPairBasis(1)
        θ = Float32(π/3)
        phase_cpu = phaseshift(basis, θ)
        phase_gpu = phaseshift(basis, θ) |> gpu
        @test Array(phase_gpu.disp) ≈ Float32.(phase_cpu.disp)
        @test Array(phase_gpu.symplectic) ≈ Float32.(phase_cpu.symplectic) rtol=1e-6
        @test device(phase_gpu) == :gpu
    end
    
    @testset "GPU Beam Splitter" begin
        basis = QuadPairBasis(2)
        transmit = 0.7f0
        bs_cpu = beamsplitter(basis, transmit)
        bs_gpu = beamsplitter(basis, transmit) |> gpu
        @test Array(bs_gpu.disp) ≈ Float32.(bs_cpu.disp)
        @test Array(bs_gpu.symplectic) ≈ Float32.(bs_cpu.symplectic) rtol=1e-6
        @test device(bs_gpu) == :gpu
    end
    
    @testset "GPU Two-Mode Squeeze" begin
        basis = QuadPairBasis(2)
        r, θ = 0.2f0, Float32(π/6)
        twosq_cpu = twosqueeze(basis, r, θ)
        twosq_gpu = twosqueeze(basis, r, θ) |> gpu
        @test Array(twosq_gpu.disp) ≈ Float32.(twosq_cpu.disp)
        @test Array(twosq_gpu.symplectic) ≈ Float32.(twosq_cpu.symplectic) rtol=1e-6
        @test device(twosq_gpu) == :gpu
    end
end

@testitem "GPU Foundation - Channel Operations" begin
    using Gabs
    using Gabs: device
    using LinearAlgebra
    using CUDA        
        
    @testset "GPU Attenuator Channel" begin
        basis = QuadPairBasis(1)
        θ, n = Float32(π/6), 2.0f0
        att_cpu = attenuator(basis, θ, n)
        att_gpu = attenuator(basis, θ, n) |> gpu
        @test att_gpu.disp isa CuVector{Float32}
        @test att_gpu.transform isa CuMatrix{Float32}
        @test att_gpu.noise isa CuMatrix{Float32}
        @test device(att_gpu) == :gpu
        @test Array(att_gpu.disp) ≈ Float32.(att_cpu.disp)
        @test Array(att_gpu.transform) ≈ Float32.(att_cpu.transform) rtol=1e-6
        @test Array(att_gpu.noise) ≈ Float32.(att_cpu.noise) rtol=1e-6
        coh_gpu = coherentstate(basis, 1.0f0) |> gpu
        attenuated_gpu = att_gpu * coh_gpu
        coh_cpu = coherentstate(basis, 1.0f0)
        attenuated_cpu = att_cpu * coh_cpu
        @test Array(attenuated_gpu.mean) ≈ Float32.(attenuated_cpu.mean) rtol=1e-6
        @test Array(attenuated_gpu.covar) ≈ Float32.(attenuated_cpu.covar) rtol=1e-5
    end
    
    @testset "GPU Amplifier Channel" begin
        basis = QuadPairBasis(1)
        r, n = 0.2f0, 1.5f0
        amp_cpu = amplifier(basis, r, n)
        amp_gpu = amplifier(basis, r, n) |> gpu
        @test Array(amp_gpu.disp) ≈ Float32.(amp_cpu.disp)
        @test Array(amp_gpu.transform) ≈ Float32.(amp_cpu.transform) rtol=1e-6
        @test Array(amp_gpu.noise) ≈ Float32.(amp_cpu.noise) rtol=1e-5
        @test device(amp_gpu) == :gpu
    end
end

@testitem "GPU Foundation - Wigner Functions" begin
    using Gabs
    using Gabs: device
    using LinearAlgebra
    using CUDA
        
    @testset "GPU Single-Point Wigner" begin
        basis = QuadPairBasis(1)
        vac_cpu = vacuumstate(basis)
        vac_gpu = vacuumstate(basis) |> gpu
        coh_cpu = coherentstate(basis, 1.0f0)
        coh_gpu = coherentstate(basis, 1.0f0) |> gpu
        test_points = [[0.0f0, 0.0f0], [1.0f0, 0.5f0], [-0.5f0, 1.0f0]]
        for x in test_points
            w_cpu_vac = wigner(vac_cpu, x)
            w_gpu_vac = wigner(vac_gpu, x)
            @test w_gpu_vac ≈ w_cpu_vac rtol=1e-5
            w_cpu_coh = wigner(coh_cpu, Float32.(x))
            w_gpu_coh = wigner(coh_gpu, x)
            @test w_gpu_coh ≈ w_cpu_coh rtol=1e-5
        end
    end
    
    @testset "GPU Batch Wigner Evaluation" begin
        basis = QuadPairBasis(1)
        coh_gpu = coherentstate(basis, 1.0f0) |> gpu
        n_points = 100
        x_points = CuMatrix{Float32}(randn(Float32, 2, n_points))
        w_batch = wigner(coh_gpu, x_points)
        @test w_batch isa CuVector{Float32}
        @test length(w_batch) == n_points
        @test all(isfinite.(Array(w_batch)))
        coh_cpu = coherentstate(basis, 1.0f0)
        for i in 1:min(10, n_points)
            x_point = Array(x_points[:, i])
            w_individual = wigner(coh_cpu, x_point)
            w_batch_individual = Array(w_batch)[i]
            @test w_batch_individual ≈ w_individual rtol=1e-4
        end
    end
    
    @testset "GPU Single-Point Wigner Characteristic" begin
        basis = QuadPairBasis(1)
        vac_cpu = vacuumstate(basis)
        vac_gpu = vacuumstate(basis) |> gpu
        test_points = [[0.0f0, 0.0f0], [0.5f0, 0.3f0], [-0.2f0, 0.8f0]]
        for xi in test_points
            chi_cpu = wignerchar(vac_cpu, xi)
            chi_gpu = wignerchar(vac_gpu, xi)
            @test real(chi_gpu) ≈ real(chi_cpu) rtol=1e-5
            @test imag(chi_gpu) ≈ imag(chi_cpu) rtol=1e-5
        end
    end
    
    @testset "GPU Batch Wigner Characteristic" begin
        basis = QuadPairBasis(1)
        sq_gpu = squeezedstate(basis, 0.3f0, Float32(π/4)) |> gpu
        n_points = 50
        xi_points = CuMatrix{Float32}(randn(Float32, 2, n_points) * 0.5f0)
        chi_batch = wignerchar(sq_gpu, xi_points)
        @test chi_batch isa CuVector{ComplexF32}
        @test length(chi_batch) == n_points
        @test all(isfinite.(real(Array(chi_batch))))
        @test all(isfinite.(imag(Array(chi_batch))))
    end
    
    @testset "GPU Wigner Grid Utility" begin
        basis = QuadPairBasis(1)
        coh_gpu = coherentstate(basis, 0.5f0) |> gpu
        x_range = (-2.0f0, 2.0f0)
        p_range = (-2.0f0, 2.0f0)
        nx, np = 20, 25
        x_vals = collect(range(x_range[1], x_range[2], length=nx))
        p_vals = collect(range(p_range[1], p_range[2], length=np))
        x_grid = repeat(x_vals, 1, np)[:]
        p_grid = repeat(p_vals', nx, 1)[:]
        points = CuArray(Float32.([x_grid'; p_grid']))
        w_grid = wigner(coh_gpu, points)
        @test length(w_grid) == nx * np
        @test all(Array(w_grid) .> 0)
    end
end

@testitem "GPU Foundation - Tensor Products and Partial Traces" begin
    using Gabs
    using Gabs: device
    using LinearAlgebra
    using CUDA  
        
    @testset "GPU State Tensor Products" begin
        basis1 = QuadPairBasis(1)
        basis2 = QuadPairBasis(1)
        coh1_gpu = coherentstate(basis1, 1.0f0) |> gpu
        vac2_gpu = vacuumstate(basis2) |> gpu
        coh1_cpu = coherentstate(basis1, 1.0f0)
        vac2_cpu = vacuumstate(basis2)
        tensor_gpu = tensor(coh1_gpu, vac2_gpu)
        tensor_cpu = tensor(coh1_cpu, vac2_cpu)
        @test tensor_gpu.basis.nmodes == 2
        @test device(tensor_gpu) == :gpu
        @test Array(tensor_gpu.mean) ≈ Float32.(tensor_cpu.mean) rtol=1e-6
        @test Array(tensor_gpu.covar) ≈ Float32.(tensor_cpu.covar) rtol=1e-6
    end
    
    @testset "Mixed Device Tensor Products" begin
        basis1 = QuadPairBasis(1)
        coh_cpu = coherentstate(basis1, 1.0f0)
        vac_gpu = vacuumstate(basis1) |> gpu
        tensor_mixed = tensor(coh_cpu, vac_gpu)
        @test device(tensor_mixed) == :gpu
    end
    
    @testset "GPU Unitary Tensor Products" begin
        basis1 = QuadPairBasis(1)
        disp1_gpu = displace(basis1, 0.5f0) |> gpu
        disp2_gpu = displace(basis1, -0.3f0) |> gpu
        disp1_cpu = displace(basis1, 0.5f0)
        disp2_cpu = displace(basis1, -0.3f0)
        tensor_gpu = tensor(disp1_gpu, disp2_gpu)
        tensor_cpu = tensor(disp1_cpu, disp2_cpu)
        @test device(tensor_gpu) == :gpu
        @test Array(tensor_gpu.disp) ≈ Float32.(tensor_cpu.disp) rtol=1e-6
        @test Array(tensor_gpu.symplectic) ≈ Float32.(tensor_cpu.symplectic) rtol=1e-6
    end
    
    @testset "GPU Partial Traces" begin
        basis = QuadPairBasis(2)
        epr_gpu = eprstate(basis, 0.3f0, Float32(π/4)) |> gpu
        epr_cpu = eprstate(basis, 0.3f0, Float32(π/4))
        traced_gpu = ptrace(epr_gpu, [1])
        traced_cpu = ptrace(epr_cpu, [1])
        @test traced_gpu.basis.nmodes == 1
        @test device(traced_gpu) == :gpu
        @test Array(traced_gpu.mean) ≈ Float32.(traced_cpu.mean) rtol=1e-6
        @test Array(traced_gpu.covar) ≈ Float32.(traced_cpu.covar) rtol=1e-5
    end
end

@testitem "GPU Foundation - Device Management" begin
    using Gabs
    using Gabs: device
    using LinearAlgebra
     using CUDA
        
    @testset "Device Transfer Functions" begin
        basis = QuadPairBasis(1)
        vac_cpu = vacuumstate(basis)
        @test device(vac_cpu) == :cpu
        vac_gpu = vac_cpu |> gpu
        @test device(vac_gpu) == :gpu
        @test vac_gpu.mean isa CuVector{Float32}
        vac_back = vac_gpu |> cpu
        @test device(vac_back) == :cpu
        @test vac_back.mean isa Vector{Float32}
        disp_cpu = displace(basis, 1.0)
        disp_gpu = disp_cpu |> gpu
        @test device(disp_gpu) == :gpu
        disp_back = disp_gpu |> cpu
        @test device(disp_back) == :cpu
        att_cpu = attenuator(basis, π/4, 1.0)
        att_gpu = att_cpu |> gpu
        @test device(att_gpu) == :gpu
        att_back = att_gpu |> cpu
        @test device(att_back) == :cpu
    end
    
    @testset "Precision Control" begin
        basis = QuadPairBasis(1)
        vac = vacuumstate(basis)
        vac_f32 = gpu(vac, precision=Float32)
        vac_f64 = gpu(vac, precision=Float64)
        @test eltype(vac_f32.mean) == Float32
        @test eltype(vac_f64.mean) == Float64
        disp = displace(basis, 1.0)
        disp_f32 = gpu(disp, precision=Float32)
        disp_f64 = gpu(disp, precision=Float64)
        @test eltype(disp_f32.disp) == Float32
        @test eltype(disp_f64.disp) == Float64
    end
    
    @testset "Array Device Functions" begin
        x_cpu = randn(10)
        x_gpu = x_cpu |> gpu
        @test x_gpu isa CuVector{Float32}
        @test device(x_gpu) == :gpu
        x_back = x_gpu |> cpu
        @test x_back isa Vector{Float32}
        @test device(x_back) == :cpu
        @test x_back ≈ Float32.(x_cpu)
    end
end

#tensor tests
@testitem "GPU Tensor Products - Operators" begin
    using CUDA
    using Gabs
    using Gabs: device
    using LinearAlgebra
    
    @testset "GPU GaussianUnitary Tensor Products" begin
        basis1 = QuadPairBasis(1)
        basis2 = QuadPairBasis(1)
        disp1_cpu = displace(basis1, 0.5f0)
        disp2_cpu = displace(basis2, -0.3f0)
        disp1_gpu = disp1_cpu |> gpu
        disp2_gpu = disp2_cpu |> gpu
        @test device(disp1_gpu) == :gpu
        @test device(disp2_gpu) == :gpu
        tensor_gpu = tensor(disp1_gpu, disp2_gpu)
        tensor_cpu = tensor(disp1_cpu, disp2_cpu)
        @test device(tensor_gpu) == :gpu
        @test tensor_gpu.basis.nmodes == 2
        @test Array(tensor_gpu.disp) ≈ Float32.(tensor_cpu.disp) rtol=1e-6
        @test Array(tensor_gpu.symplectic) ≈ Float32.(tensor_cpu.symplectic) rtol=1e-6
        tensor_typed = tensor(CuVector{Float64}, CuMatrix{Float64}, disp1_gpu, disp2_gpu)
        @test eltype(tensor_typed.disp) == Float64
        @test eltype(tensor_typed.symplectic) == Float64
        vac1_gpu = vacuumstate(basis1) |> gpu
        vac2_gpu = vacuumstate(basis2) |> gpu
        tensor_state_gpu = tensor(vac1_gpu, vac2_gpu)
        result_gpu = tensor_gpu * tensor_state_gpu
        vac1_cpu = vacuumstate(basis1)
        vac2_cpu = vacuumstate(basis2)  
        tensor_state_cpu = tensor(vac1_cpu, vac2_cpu)
        result_cpu = tensor_cpu * tensor_state_cpu
        @test Array(result_gpu.mean) ≈ Float32.(result_cpu.mean) rtol=1e-5
        @test Array(result_gpu.covar) ≈ Float32.(result_cpu.covar) rtol=1e-5
    end
    
    @testset "GPU GaussianChannel Tensor Products" begin
        basis1 = QuadPairBasis(1)
        basis2 = QuadPairBasis(1)
        att1_cpu = attenuator(basis1, π/6, 1.5)
        att2_cpu = attenuator(basis2, π/4, 2.0)
        att1_gpu = att1_cpu |> gpu
        att2_gpu = att2_cpu |> gpu
        @test device(att1_gpu) == :gpu
        @test device(att2_gpu) == :gpu
        tensor_gpu = tensor(att1_gpu, att2_gpu)
        tensor_cpu = tensor(att1_cpu, att2_cpu)
        @test device(tensor_gpu) == :gpu
        @test tensor_gpu.basis.nmodes == 2
        @test Array(tensor_gpu.disp) ≈ Float32.(tensor_cpu.disp) rtol=1e-6
        @test Array(tensor_gpu.transform) ≈ Float32.(tensor_cpu.transform) rtol=1e-5
        @test Array(tensor_gpu.noise) ≈ Float32.(tensor_cpu.noise) rtol=1e-5
        tensor_typed = tensor(CuVector{Float64}, CuMatrix{Float64}, att1_gpu, att2_gpu)
        @test eltype(tensor_typed.disp) == Float64
        @test eltype(tensor_typed.transform) == Float64
        @test eltype(tensor_typed.noise) == Float64
    end
    
    @testset "Mixed Operator Types" begin
        basis = QuadPairBasis(1)
        disp_gpu = displace(basis, 1.0f0) |> gpu
        att_gpu = attenuator(basis, π/4, 1.0) |> gpu
        @test_throws ArgumentError tensor(disp_gpu, att_gpu)
    end
    
    @testset "Error Handling" begin
        basis1 = QuadPairBasis(1)  
        basis2 = QuadBlockBasis(1)
        disp1_gpu = displace(basis1, 1.0f0) |> gpu
        disp2_gpu = displace(basis2, 1.0f0) |> gpu
        @test_throws ArgumentError tensor(disp1_gpu, disp2_gpu)
        disp1_h1 = displace(basis1, 1.0f0, ħ=1) |> gpu  
        disp1_h2 = displace(basis1, 1.0f0, ħ=2) |> gpu
        @test_throws ArgumentError tensor(disp1_h1, disp1_h2)
    end
end

@testitem "GPU Linear Combinations - Core Functionality" begin
    using CUDA
    using Gabs
    using Gabs: device, GaussianLinearCombination
    using LinearAlgebra

    @testset "Device Management & Transfer" begin
        basis = QuadPairBasis(1)
        state1 = coherentstate(basis, 1.0f0 + 0.5f0im)
        state2 = coherentstate(basis, -1.0f0 + 0.3f0im)
        coeffs = [0.6f0, 0.8f0]
        lc_cpu = GaussianLinearCombination(basis, coeffs, [state1, state2])
        @test device(lc_cpu) == :cpu
        @test lc_cpu.coeffs isa Vector{Float32}
        @test device(lc_cpu.states[1]) == :cpu
        lc_gpu = lc_cpu |> gpu
        @test device(lc_gpu) == :gpu
        @test lc_gpu.coeffs isa Vector{Float32}
        @test device(lc_gpu.states[1]) == :gpu
        @test device(lc_gpu.states[2]) == :gpu
        @test lc_gpu.states[1].mean isa CuVector{Float32}
        @test lc_gpu.states[1].covar isa CuMatrix{Float32}
        @test lc_gpu.coeffs ≈ lc_cpu.coeffs
        lc_back = lc_gpu |> cpu
        @test device(lc_back) == :cpu
        @test lc_back.coeffs ≈ lc_cpu.coeffs
        @test Array(lc_gpu.states[1].mean) ≈ lc_back.states[1].mean rtol=1e-6
        lc_gpu_f64 = gpu(lc_cpu, precision=Float64)
        @test eltype(lc_gpu_f64.coeffs) == Float64
        @test lc_gpu_f64.states[1].mean isa CuVector{Float64}
    end
    
    @testset "Creating GPU Linear Combinations Directly" begin
        basis = QuadPairBasis(1)
        state1_cpu = coherentstate(basis, 1.0f0)
        state2_cpu = squeezedstate(basis, 0.3f0, Float32(π/4))
        lc1 = GaussianLinearCombination(basis, [0.7f0, 0.3f0], [state1_cpu, state2_cpu]) |> gpu
        @test device(lc1) == :gpu
        state1_gpu = coherentstate(basis, 1.0f0) |> gpu
        state2_gpu = squeezedstate(basis, 0.3f0, Float32(π/4)) |> gpu
        lc2 = GaussianLinearCombination(basis, [0.7f0, 0.3f0], [state1_gpu, state2_gpu])
        @test device(lc2) == :gpu
        @test lc1.coeffs ≈ lc2.coeffs
        @test Array(lc1.states[1].mean) ≈ Array(lc2.states[1].mean) rtol=1e-6
    end

    @testset "Arithmetic Operations" begin
        basis = QuadPairBasis(1)
        state1 = coherentstate(basis, 1.0f0) |> gpu
        state2 = coherentstate(basis, -1.0f0) |> gpu
        state3 = squeezedstate(basis, 0.2f0, 0.0f0) |> gpu
        lc1 = GaussianLinearCombination(basis, [0.6f0, 0.4f0], [state1, state2])
        lc2 = GaussianLinearCombination(basis, [0.3f0], [state3])
        @test device(lc1) == :gpu
        @test device(lc2) == :gpu
        lc_sum = lc1 + lc2
        @test length(lc_sum) == 3
        @test device(lc_sum) == :gpu
        @test lc_sum.coeffs ≈ [0.6f0, 0.4f0, 0.3f0]
        lc_diff = lc1 - lc2
        @test length(lc_diff) == 3
        @test device(lc_diff) == :gpu
        @test lc_diff.coeffs ≈ [0.6f0, 0.4f0, -0.3f0]
        lc_scaled = 2.0f0 * lc1
        @test length(lc_scaled) == 2
        @test device(lc_scaled) == :gpu
        @test lc_scaled.coeffs ≈ [1.2f0, 0.8f0]
        lc_scaled2 = lc1 * 0.5f0
        @test lc_scaled2.coeffs ≈ [0.3f0, 0.2f0]
        lc_neg = -lc1
        @test lc_neg.coeffs ≈ [-0.6f0, -0.4f0]
        @test device(lc_neg) == :gpu
    end

    @testset "Normalization" begin
        basis = QuadPairBasis(1)
        state1 = vacuumstate(basis) |> gpu
        state2 = coherentstate(basis, 1.0f0) |> gpu
        coeffs = [3.0f0, 4.0f0]
        lc = GaussianLinearCombination(basis, coeffs, [state1, state2])
        @test device(lc) == :gpu
        initial_norm = sqrt(sum(abs2, lc.coeffs))
        @test initial_norm ≈ 5.0f0
        Gabs.normalize!(lc)
        @test device(lc) == :gpu
        final_norm = sqrt(sum(abs2, lc.coeffs))
        @test final_norm ≈ 1.0f0 rtol=1e-6
        @test lc.coeffs ≈ [0.6f0, 0.8f0] rtol=1e-6
        lc_zero = GaussianLinearCombination(basis, [0.0f0, 0.0f0], [state1, state2])
        Gabs.normalize!(lc_zero)
        @test lc_zero.coeffs == [0.0f0, 0.0f0]
    end

    @testset "Simplification" begin
        basis = QuadPairBasis(1)
        vac = vacuumstate(basis) |> gpu
        coh = coherentstate(basis, 1.0f0) |> gpu
        coeffs1 = CuArray([0.9f0, Float32(1e-15), 0.1f0])
        lc1 = GaussianLinearCombination(basis, coeffs1, [vac, coh, vac])
        @test length(lc1) == 3
        Gabs.simplify!(lc1)
        @test length(lc1) == 1
        @test device(lc1) == :gpu
        coh2 = coherentstate(basis, 1.0f0) |> gpu
        coeffs2 = CuArray([0.5f0, 0.3f0, 0.2f0])
        lc2 = GaussianLinearCombination(basis, coeffs2, [vac, coh, coh2])
        @test length(lc2) == 3
        Gabs.simplify!(lc2)
        @test length(lc2) == 2
        @test device(lc2) == :gpu
        cpu_coeffs = Array(lc2.coeffs)
        cpu_vac_mean = Array(vac.mean)
        cpu_coh_mean = Array(coh.mean)
        vac_idx = findfirst(i -> Array(lc2.states[i].mean) ≈ cpu_vac_mean, 1:length(lc2))
        coh_idx = findfirst(i -> Array(lc2.states[i].mean) ≈ cpu_coh_mean, 1:length(lc2))
        @test vac_idx !== nothing
        @test coh_idx !== nothing
        @test cpu_coeffs[vac_idx] ≈ 0.5f0
        @test cpu_coeffs[coh_idx] ≈ 0.5f0
        coeffs3 = CuArray([Float32(1e-16), Float32(1e-17)])
        lc3 = GaussianLinearCombination(basis, coeffs3, [vac, coh])
        Gabs.simplify!(lc3)
        @test length(lc3) == 1
        @test device(lc3) == :gpu
    end

    @testset "GPU Operations - Gaussian Unitaries" begin
        basis = QuadPairBasis(1)
        state1 = coherentstate(basis, 1.0f0) |> gpu
        state2 = coherentstate(basis, -1.0f0) |> gpu
        lc_gpu = GaussianLinearCombination(basis, [0.7f0, 0.3f0], [state1, state2])
        @test device(lc_gpu) == :gpu
        disp_gpu = displace(basis, 0.5f0) |> gpu
        result_gpu = disp_gpu * lc_gpu
        @test device(result_gpu) == :gpu
        @test length(result_gpu) == 2
        @test result_gpu.coeffs ≈ lc_gpu.coeffs
        expected1 = disp_gpu * state1
        expected2 = disp_gpu * state2
        @test Array(result_gpu.states[1].mean) ≈ Array(expected1.mean) rtol=1e-6
        @test Array(result_gpu.states[2].mean) ≈ Array(expected2.mean) rtol=1e-6
        disp_cpu = displace(basis, 0.2f0)
        result_mixed = disp_cpu * lc_gpu
        @test device(result_mixed) == :gpu
        squeeze_gpu = squeeze(basis, 0.3f0, Float32(π/4)) |> gpu
        squeezed_lc = squeeze_gpu * lc_gpu
        @test device(squeezed_lc) == :gpu
        @test length(squeezed_lc) == 2
        phase_gpu = phaseshift(basis, π/3) |> gpu
        phase_shifted = phase_gpu * lc_gpu
        @test device(phase_shifted) == :gpu
    end

    @testset "GPU Operations - Gaussian Channels" begin
        basis = QuadPairBasis(1)
        coh1 = coherentstate(basis, 1.0f0) |> gpu
        coh2 = coherentstate(basis, 0.5f0) |> gpu
        lc_gpu = GaussianLinearCombination(basis, [0.8f0, 0.2f0], [coh1, coh2])
        att_gpu = attenuator(basis, π/6, 2.0f0) |> gpu
        attenuated = att_gpu * lc_gpu
        @test device(attenuated) == :gpu
        @test length(attenuated) == 2
        @test attenuated.coeffs ≈ lc_gpu.coeffs
        expected1 = att_gpu * coh1
        expected2 = att_gpu * coh2
        @test Array(attenuated.states[1].mean) ≈ Array(expected1.mean) rtol=1e-6
        @test Array(attenuated.states[2].covar) ≈ Array(expected2.covar) rtol=1e-5
        amp_gpu = amplifier(basis, 0.2f0, 1.5f0) |> gpu
        amplified = amp_gpu * lc_gpu
        @test device(amplified) == :gpu
        att_cpu = attenuator(basis, π/8, 1.0f0)
        result_mixed = att_cpu * lc_gpu
        @test device(result_mixed) == :gpu
    end

    @testset "State Metrics" begin
        basis = QuadPairBasis(1)
        state1 = coherentstate(basis, 1.0f0) |> gpu
        state2 = squeezedstate(basis, 0.3f0, Float32(π/4)) |> gpu
        lc_gpu = GaussianLinearCombination(basis, [0.6f0, 0.8f0], [state1, state2])
        @test device(lc_gpu) == :gpu
        p = purity(lc_gpu)
        @test p == 1.0
        s = entropy_vn(lc_gpu)
        @test s == 0.0
        basis_multi = QuadPairBasis(2)
        epr = eprstate(basis_multi, 0.5f0, Float32(π/3)) |> gpu
        lc_multi = GaussianLinearCombination(basis_multi, [1.0f0], [epr])
        @test purity(lc_multi) == 1.0
        @test entropy_vn(lc_multi) == 0.0
    end

    @testset "Mixed Device Operations" begin
        basis = QuadPairBasis(1)
        state1_cpu = coherentstate(basis, 1.0f0)
        state2_cpu = coherentstate(basis, -1.0f0)
        lc_cpu = GaussianLinearCombination(basis, [0.7f0, 0.3f0], [state1_cpu, state2_cpu])
        state3_gpu = squeezedstate(basis, 0.2f0, 0.0f0) |> gpu
        lc_gpu = GaussianLinearCombination(basis, [0.5f0], [state3_gpu])
        @test device(lc_cpu) == :cpu
        @test device(lc_gpu) == :gpu
        lc_sum = lc_cpu + lc_gpu
        disp_cpu = displace(basis, 0.3f0)
        disp_gpu = displace(basis, 0.4f0) |> gpu
        result1 = disp_cpu * lc_gpu
        @test device(result1) == :gpu
        result2 = disp_gpu * lc_cpu
        @test device(result2) == :gpu
    end

    @testset "Error Handling & Edge Cases" begin
        basis1 = QuadPairBasis(1)
        basis2 = QuadBlockBasis(1)
        state1 = coherentstate(basis1, 1.0f0) |> gpu
        state2 = coherentstate(basis2, 1.0f0) |> gpu
        lc1 = GaussianLinearCombination(basis1, [0.5f0], [state1])
        lc2 = GaussianLinearCombination(basis2, [0.5f0], [state2])
        @test_throws ArgumentError lc1 + lc2
        state3 = GaussianState(basis1, CuArray([0.0f0, 0.0f0]), CuMatrix{Float32}(I(2)); ħ = 4)
        lc3 = GaussianLinearCombination(basis1, [1.0f0], [state3])
        @test_throws ArgumentError lc1 + lc3
        disp_wrong = displace(basis2, 1.0f0) |> gpu
        @test_throws ArgumentError disp_wrong * lc1
        @test_throws ArgumentError GaussianLinearCombination(basis1, Float32[], typeof(state1)[])
        tiny_coeffs = CuArray([Float32(1e-20), Float32(1e-21)])
        tiny_states = [state1, coherentstate(basis1, 2.0f0) |> gpu]
        lc_tiny = GaussianLinearCombination(basis1, tiny_coeffs, tiny_states)
        Gabs.simplify!(lc_tiny)
        @test length(lc_tiny) == 1
        @test device(lc_tiny) == :gpu
        if CUDA.functional()
            @test true
        else
            @warn "CUDA not functional, testing fallback behavior"
        end
    end

    @testset "Type Stability & Performance" begin
        basis = QuadPairBasis(1)
        state1 = coherentstate(basis, 1.0f0) |> gpu
        state2 = squeezedstate(basis, 0.3f0, Float32(π/4)) |> gpu
        lc = GaussianLinearCombination(basis, [0.6f0, 0.8f0], [state1, state2])
        @inferred device(lc)
        @inferred purity(lc)
        @inferred entropy_vn(lc)
        disp = displace(basis, 0.5f0) |> gpu
        result = disp * lc
        @test result.states[1].mean isa CuVector{Float32}
        @test result.states[1].covar isa CuMatrix{Float32}
        lc_scaled = 2.0f0 * lc
        @test lc_scaled.coeffs isa Vector{Float32}
        @test lc_scaled.states[1].mean isa CuVector{Float32}
        lc_copy = GaussianLinearCombination(basis, copy(lc.coeffs), copy(lc.states))
        Gabs.normalize!(lc_copy)
        @test lc_copy.coeffs isa Vector{Float32}
        @test device(lc_copy) == :gpu
    end

    @testset "Multi-Mode GPU Linear Combinations" begin
        basis = QuadPairBasis(2)
        α_vec1 = [1.0f0 + 0.5f0im, -0.3f0 + 0.8f0im]
        α_vec2 = [0.5f0, 0.2f0 - 0.4f0im]
        coh1 = coherentstate(basis, α_vec1) |> gpu
        coh2 = coherentstate(basis, α_vec2) |> gpu
        epr = eprstate(basis, 0.3f0, Float32(π/6)) |> gpu
        coeffs = [0.5f0, 0.3f0, 0.2f0]
        lc_multi = GaussianLinearCombination(basis, coeffs, [coh1, coh2, epr])
        @test device(lc_multi) == :gpu
        @test length(lc_multi) == 3
        @test lc_multi.basis.nmodes == 2
        squeeze_op = squeeze(basis, [0.2f0, 0.3f0], [0.0f0, Float32(π/4)]) |> gpu
        squeezed_multi = squeeze_op * lc_multi
        @test device(squeezed_multi) == :gpu
        @test length(squeezed_multi) == 3
        @test squeezed_multi.coeffs ≈ coeffs
        bs = beamsplitter(basis, 0.7f0) |> gpu
        bs_result = bs * lc_multi
        @test device(bs_result) == :gpu
        twosq = twosqueeze(basis, 0.2f0, Float32(π/3)) |> gpu
        twosq_result = twosq * lc_multi
        @test device(twosq_result) == :gpu
    end

    @testset "Professional API Validation" begin
        @info "=== Testing Professional GPU API for Linear Combinations ==="
        basis = QuadPairBasis(1)
        state1 = coherentstate(basis, 1.0f0) |> gpu
        state2 = squeezedstate(basis, 0.3f0, Float32(π/4)) |> gpu
        lc = GaussianLinearCombination(basis, [0.7f0, 0.3f0], [state1, state2])
        op = displace(basis, 0.5f0) |> gpu
        result = op * lc
        @test device(lc) == :gpu
        @test device(result) == :gpu
        @test result.coeffs ≈ lc.coeffs
        α_gpu = CuArray([1.0f0 + 0.5f0im])
        auto_state = coherentstate(basis, α_gpu)
        lc_auto = GaussianLinearCombination(basis, [1.0f0], [auto_state])
        @test device(lc_auto) == :gpu
        cpu_op = squeeze(basis, 0.2f0, 0.0f0)
        mixed_result = cpu_op * lc
        @test device(mixed_result) == :gpu
    end
end

@testitem "GPU Linear Combinations Further Coverage" begin
    using CUDA
    using Gabs
    using Gabs: device, GaussianLinearCombination
    using LinearAlgebra
    using Test

    @testset "Basic GPU LC Creation and Device Detection" begin
        basis = QuadPairBasis(1)
        cpu_state = coherentstate(basis, 1.0f0)
        cpu_lc = GaussianLinearCombination(basis, [0.7f0], [cpu_state])
        gpu_lc = cpu_lc |> gpu
        @test device(cpu_lc) == :cpu
        @test device(gpu_lc) == :gpu
        @test gpu_lc.coeffs isa Vector{Float32}
        @test Gabs._is_gpu_array(gpu_lc.coeffs) == false
        @test gpu_lc.states[1].mean isa CuVector{Float32}
        gpu_state = coherentstate(basis, 1.0f0) |> gpu
        direct_gpu_lc = GaussianLinearCombination(basis, [0.8f0], [gpu_state])
        @test device(direct_gpu_lc) == :gpu
        @test direct_gpu_lc.coeffs isa Vector{Float32}
        cpu_state2 = squeezedstate(basis, 0.3f0, Float32(π/4))
        gpu_state2 = cpu_state2 |> gpu
        mixed_lc = GaussianLinearCombination(basis, [0.6f0, 0.4f0], [cpu_state, gpu_state2])
        @test device(mixed_lc) == :cpu
        full_gpu_lc = mixed_lc |> gpu
        @test device(full_gpu_lc) == :gpu
        @test all(device(s) == :gpu for s in full_gpu_lc.states)
    end

    @testset "Arithmetic Operations - Comprehensive Device Promotion" begin
        basis = QuadPairBasis(1)
        cpu_state1 = coherentstate(basis, 1.0f0)
        cpu_state2 = coherentstate(basis, -1.0f0)
        gpu_state1 = coherentstate(basis, 0.5f0) |> gpu
        gpu_state2 = squeezedstate(basis, 0.2f0, 0.0f0) |> gpu
        cpu_lc = GaussianLinearCombination(basis, [0.7f0, 0.3f0], [cpu_state1, cpu_state2])
        gpu_lc = GaussianLinearCombination(basis, [0.6f0, 0.4f0], [gpu_state1, gpu_state2])
        result1 = cpu_lc + gpu_lc
        @test device(result1) == :gpu
        @test length(result1) == 4
        @test result1.coeffs isa Vector{Float32}
        @test all(device(s) == :gpu for s in result1.states)
        result2 = gpu_lc + cpu_lc  
        @test device(result2) == :gpu
        @test length(result2) == 4
        cpu_lc2 = GaussianLinearCombination(basis, [0.5f0], [cpu_state1])
        result3 = cpu_lc + cpu_lc2
        @test device(result3) == :cpu
        @test length(result3) == 3
        gpu_lc2 = GaussianLinearCombination(basis, [0.8f0], [gpu_state1])
        result4 = gpu_lc + gpu_lc2
        @test device(result4) == :gpu
        @test length(result4) == 3
        scaled_cpu = 2.0f0 * cpu_lc
        scaled_gpu = 2.0f0 * gpu_lc
        @test device(scaled_cpu) == :cpu
        @test device(scaled_gpu) == :gpu
        @test scaled_gpu.coeffs ≈ [1.2f0, 0.8f0]
        result5 = cpu_state1 + gpu_lc
        @test device(result5) == :gpu
        @test length(result5) == 3
        result6 = gpu_state1 + cpu_lc
        @test device(result6) == :gpu
        result7 = cpu_lc - gpu_lc
        @test device(result7) == :gpu
        result8 = gpu_state1 - cpu_lc
        @test device(result8) == :gpu
    end

    @testset "GPU Operator Applications - All Combinations" begin
        basis = QuadPairBasis(1)
        cpu_state1 = coherentstate(basis, 1.0f0)
        cpu_state2 = squeezedstate(basis, 0.2f0, Float32(π/4))
        gpu_lc = GaussianLinearCombination(basis, [0.7f0, 0.3f0], [cpu_state1, cpu_state2]) |> gpu
        cpu_lc = GaussianLinearCombination(basis, [0.6f0, 0.4f0], [cpu_state1, cpu_state2])
        gpu_disp = displace(basis, 0.5f0) |> gpu
        result1 = gpu_disp * gpu_lc
        @test device(result1) == :gpu
        @test result1.coeffs ≈ gpu_lc.coeffs
        @test length(result1) == 2
        cpu_disp = displace(basis, 0.3f0)
        result2 = cpu_disp * gpu_lc
        @test device(result2) == :gpu
        @test result2.coeffs ≈ gpu_lc.coeffs
        result3 = gpu_disp * cpu_lc
        @test device(result3) == :gpu
        @test result3.coeffs ≈ cpu_lc.coeffs
        result4 = cpu_disp * cpu_lc
        @test device(result4) == :cpu
        gpu_squeeze = squeeze(basis, 0.2f0, Float32(π/3)) |> gpu
        gpu_phase = phaseshift(basis, Float32(π/6)) |> gpu
        chained = gpu_phase * (gpu_squeeze * gpu_lc)
        @test device(chained) == :gpu
        @test chained.coeffs ≈ gpu_lc.coeffs
        gpu_atten = attenuator(basis, π/4, 2.0f0) |> gpu
        cpu_amp = amplifier(basis, 0.1f0, 1.5f0)
        channel_result1 = gpu_atten * gpu_lc
        @test device(channel_result1) == :gpu
        channel_result2 = cpu_amp * gpu_lc
        @test device(channel_result2) == :gpu
        channel_result3 = gpu_atten * cpu_lc
        @test device(channel_result3) == :gpu
        cpu_result = cpu_disp * cpu_lc
        gpu_equivalent = cpu_disp * (cpu_lc |> gpu)
        cpu_back = gpu_equivalent |> cpu
        @test cpu_back.coeffs ≈ cpu_result.coeffs rtol=1e-6
        @test cpu_back.states[1].mean ≈ cpu_result.states[1].mean rtol=1e-6
        @test cpu_back.states[1].covar ≈ cpu_result.states[1].covar rtol=1e-5
    end

    @testset "Memory Management and Efficiency" begin
        basis = QuadPairBasis(1)
        states = [coherentstate(basis, Float32(i)) for i in 1:10]
        coeffs = Float32.(rand(10))
        cpu_lc = GaussianLinearCombination(basis, coeffs, states)
        gpu_lc = cpu_lc |> gpu
        @test sizeof(gpu_lc.coeffs) == sizeof(coeffs)
        @test gpu_lc.coeffs isa Vector{Float32}
        large_states = [coherentstate(basis, Float32(i)*0.1f0) for i in 1:100]
        large_coeffs = Float32.(randn(100))
        large_lc = GaussianLinearCombination(basis, large_coeffs, large_states)
        large_gpu_lc = large_lc |> gpu
        @test device(large_gpu_lc) == :gpu
        @test length(large_gpu_lc) == 100
        @test large_gpu_lc.coeffs isa Vector{Float32}
        transferred_back = large_gpu_lc |> cpu
        retransferred = transferred_back |> gpu
        @test device(transferred_back) == :cpu
        @test device(retransferred) == :gpu
        @test transferred_back.coeffs ≈ large_lc.coeffs
    end

    @testset "Normalization and Simplification on GPU" begin
        basis = QuadPairBasis(1)
        state1 = coherentstate(basis, 1.0f0) |> gpu
        state2 = squeezedstate(basis, 0.3f0, Float32(π/4)) |> gpu
        unnormalized_lc = GaussianLinearCombination(basis, [3.0f0, 4.0f0], [state1, state2])
        @test device(unnormalized_lc) == :gpu
        initial_norm = sqrt(sum(abs2, unnormalized_lc.coeffs))
        @test initial_norm ≈ 5.0f0
        Gabs.normalize!(unnormalized_lc)
        @test device(unnormalized_lc) == :gpu
        final_norm = sqrt(sum(abs2, unnormalized_lc.coeffs))
        @test final_norm ≈ 1.0f0 rtol=1e-6
        @test unnormalized_lc.coeffs ≈ [0.6f0, 0.8f0] rtol=1e-6
        small_coeffs = [0.9f0, Float32(1e-15), 0.1f0]
        small_states = [state1, state2, coherentstate(basis, 2.0f0) |> gpu]
        to_simplify = GaussianLinearCombination(basis, small_coeffs, small_states)
        @test length(to_simplify) == 3
        @test device(to_simplify) == :gpu
        Gabs.simplify!(to_simplify)
        @test device(to_simplify) == :gpu
        @test length(to_simplify) == 2
        identical_state = coherentstate(basis, 1.5f0) |> gpu
        combine_coeffs = [0.3f0, 0.7f0, 0.2f0]
        combine_states = [state1, identical_state, identical_state]
        to_combine = GaussianLinearCombination(basis, combine_coeffs, combine_states)
        @test length(to_combine) == 3
        Gabs.simplify!(to_combine)
        @test length(to_combine) == 2
        @test device(to_combine) == :gpu
        cpu_coeffs = Array(to_combine.coeffs)
        @test any(c -> isapprox(c, 0.9f0, rtol=1e-6), cpu_coeffs)
    end

    @testset "Complex Coefficients and Precision Handling" begin
        basis = QuadPairBasis(1)
        state1 = coherentstate(basis, 1.0f0) |> gpu
        state2 = squeezedstate(basis, 0.2f0, Float32(π/4)) |> gpu
        complex_coeffs = ComplexF32[0.6f0 + 0.8f0im, 0.3f0 - 0.4f0im]
        complex_lc = GaussianLinearCombination(basis, complex_coeffs, [state1, state2])
        @test device(complex_lc) == :gpu
        @test complex_lc.coeffs isa Vector{ComplexF32}
        @test eltype(complex_lc.coeffs) == ComplexF32
        float64_state = coherentstate(basis, 1.0) |> gpu
        mixed_lc = GaussianLinearCombination(basis, [0.5f0], [float64_state])
        @test device(mixed_lc) == :gpu
        precise_op = displace(basis, 1.0) |> gpu
        result = precise_op * complex_lc
        @test device(result) == :gpu
        @test result.coeffs ≈ complex_coeffs
        large_coeffs = ComplexF32[1000.0f0, 0.001f0]
        large_lc = GaussianLinearCombination(basis, large_coeffs, [state1, state2])
        Gabs.normalize!(large_lc)
        @test device(large_lc) == :gpu
        @test abs(sqrt(sum(abs2, large_lc.coeffs)) - 1.0f0) < 1e-6
    end

    @testset "Multi-Mode GPU Linear Combinations" begin
        basis2 = QuadPairBasis(2)
        basis3 = QuadPairBasis(3)
        α_vec = [1.0f0 + 0.5f0im, -0.3f0 + 0.8f0im]
        coh2mode = coherentstate(basis2, α_vec) |> gpu
        epr_state = eprstate(basis2, 0.4f0, Float32(π/3)) |> gpu
        lc_2mode = GaussianLinearCombination(basis2, [0.7f0, 0.3f0], [coh2mode, epr_state])
        @test device(lc_2mode) == :gpu
        @test lc_2mode.basis.nmodes == 2
        bs = beamsplitter(basis2, 0.6f0) |> gpu
        bs_result = bs * lc_2mode
        @test device(bs_result) == :gpu
        @test bs_result.basis.nmodes == 2
        α_vec3 = [1.0f0, 0.5f0im, -0.3f0 + 0.2f0im]
        coh3mode = coherentstate(basis3, α_vec3) |> gpu
        vac3mode = vacuumstate(basis3) |> gpu
        lc_3mode = GaussianLinearCombination(basis3, [0.8f0, 0.2f0], [coh3mode, vac3mode])
        @test device(lc_3mode) == :gpu
        @test lc_3mode.basis.nmodes == 3
        squeeze_multi = squeeze(basis3, [0.1f0, 0.2f0, 0.15f0], [0.0f0, Float32(π/4), Float32(π/2)]) |> gpu
        multi_result = squeeze_multi * lc_3mode
        @test device(multi_result) == :gpu
        @test length(multi_result) == 2
    end

    @testset "Error Handling and Edge Cases" begin
        basis1 = QuadPairBasis(1)
        basis2 = QuadBlockBasis(1)
        state1 = coherentstate(basis1, 1.0f0) |> gpu
        state2 = coherentstate(basis2, 1.0f0) |> gpu
        lc1 = GaussianLinearCombination(basis1, [0.5f0], [state1])
        lc2 = GaussianLinearCombination(basis2, [0.5f0], [state2])
        @test_throws ArgumentError lc1 + lc2
        @test_throws ArgumentError lc1 - lc2
        state_different_hbar = GaussianState(basis1, CuArray([0.0f0, 0.0f0]), 
                                           CuMatrix{Float32}(I(2)); ħ = 4)
        lc_different = GaussianLinearCombination(basis1, [1.0f0], [state_different_hbar])
        @test_throws ArgumentError lc1 + lc_different
        @test_throws ArgumentError GaussianLinearCombination(basis1, Float32[], typeof(state1)[])
        wrong_op = displace(basis2, 1.0f0) |> gpu
        @test_throws ArgumentError wrong_op * lc1
        tiny_lc = GaussianLinearCombination(basis1, [Float32(1e-20)], [state1])
        Gabs.simplify!(tiny_lc)
        @test length(tiny_lc) == 1
        @test device(tiny_lc) == :gpu
        if CUDA.functional()
            @test device(lc1) == :gpu
        else
            @warn "CUDA not functional - some GPU tests may not be meaningful"
        end
    end

    @testset "Performance and Type Stability" begin
        basis = QuadPairBasis(1)
        state1 = coherentstate(basis, 1.0f0) |> gpu
        state2 = squeezedstate(basis, 0.3f0, Float32(π/4)) |> gpu
        gpu_lc = GaussianLinearCombination(basis, [0.6f0, 0.8f0], [state1, state2])
        @inferred device(gpu_lc)
        @inferred length(gpu_lc) 
        @inferred gpu_lc[1]
        @inferred purity(gpu_lc)
        @inferred entropy_vn(gpu_lc)
        lc2 = GaussianLinearCombination(basis, [0.4f0], [state1])
        op = displace(basis, 0.5f0) |> gpu
        sum_result = gpu_lc + lc2
        mul_result = 2.0f0 * gpu_lc  
        op_result = op * gpu_lc
        @test device(sum_result) == :gpu
        @test device(mul_result) == :gpu
        @test device(op_result) == :gpu
        @test sum_result.coeffs isa Vector{Float32}
        @test mul_result.coeffs isa Vector{Float32}
        @test op_result.coeffs isa Vector{Float32}
        @test op_result.states[1].mean isa CuVector{Float32}
        @test op_result.states[1].covar isa CuMatrix{Float32}
        many_states = [coherentstate(basis, Float32(i)*0.1f0) |> gpu for i in 1:50]
        many_coeffs = Float32.(randn(50))
        big_lc = GaussianLinearCombination(basis, many_coeffs, many_states)
        batch_op = phaseshift(basis, π/3) |> gpu
        batch_result = batch_op * big_lc
        @test device(batch_result) == :gpu
        @test length(batch_result) == 50
        @test batch_result.coeffs isa Vector{Float32}
    end
    
    @testset "Integration with Existing Gabs Functionality" begin
        basis = QuadPairBasis(1)
        state1 = coherentstate(basis, 1.0f0) |> gpu
        state2 = squeezedstate(basis, 0.2f0, Float32(π/4)) |> gpu
        gpu_lc = GaussianLinearCombination(basis, [0.6f0, 0.8f0], [state1, state2])
        @test purity(gpu_lc) == 1.0
        @test entropy_vn(gpu_lc) == 0.0
        block_basis = QuadBlockBasis(1)
        pair_state = coherentstate(basis, 1.0f0) |> gpu
        block_state = changebasis(QuadBlockBasis, pair_state)
        mixed_basis_lc = GaussianLinearCombination(block_basis, [1.0f0], [block_state])
        @test device(mixed_basis_lc) == :gpu
        random_basis = QuadPairBasis(1)
        random_state = randstate(random_basis; pure=true) |> gpu
        random_lc = GaussianLinearCombination(random_basis, [1.0f0], [random_state])
        @test device(random_lc) == :gpu
        att = attenuator(basis, π/6, 2.0f0) |> gpu
        amp = amplifier(basis, 0.1f0, 1.2f0) |> gpu
        composed_result = amp * (att * gpu_lc)
        @test device(composed_result) == :gpu
        @test composed_result.coeffs ≈ gpu_lc.coeffs
        x_test = [0.0f0, 0.0f0]
    end

    @testset "Professional API Usage Examples" begin
        basis = QuadPairBasis(1)
        state1 = coherentstate(basis, 1.0f0) |> gpu
        state2 = squeezedstate(basis, 0.3f0, Float32(π/4)) |> gpu
        cat_like = GaussianLinearCombination(basis, [0.7f0, 0.3f0], [state1, state2])
        @test device(cat_like) == :gpu
        cpu_disp = displace(basis, 0.5f0)
        result = cpu_disp * cat_like
        @test device(result) == :gpu
        cpu_state = thermalstate(basis, 1.0f0)
        mixed_result = cpu_state + cat_like
        @test device(mixed_result) == :gpu
        @test cat_like.coeffs isa Vector{Float32}
        @test Gabs._is_gpu_array(cat_like.coeffs) == false
        @test cat_like.states[1].mean isa CuVector{Float32}
        ops = [squeeze(basis, Float32(0.1f0*i), Float32(π/6*i)) |> gpu for i in 1:5]
        results = [op * cat_like for op in ops]
        @test all(device(r) == :gpu for r in results)
    end
end

@testitem "GPU Basis Operations" begin
    using Gabs
    using Gabs: device, GaussianLinearCombination
    using CUDA
    using LinearAlgebra
    using Test

    if !CUDA.functional()
        @warn "CUDA not functional. Skipping GPU basis operations tests."
        return
    end

    @testset "Public API Device Management" begin
        
        @testset "Device Detection with Public API" begin
            cpu_array = rand(Float32, 10)
            gpu_array = CuArray(cpu_array)
            @test device(cpu_array) == :cpu
            @test device(gpu_array) == :gpu
            basis = QuadPairBasis(2)
            cpu_state = coherentstate(basis, 1.0 + 0.5im)
            gpu_state = gpu(cpu_state, precision=Float32)
            @test device(cpu_state) == :cpu
            @test device(gpu_state) == :gpu
            cpu_op = displace(basis, 0.5 + 0.3im)
            gpu_op = gpu(cpu_op, precision=Float32)
            @test device(cpu_op) == :cpu
            @test device(gpu_op) == :gpu
        end
        
        @testset "Device Transfer with Public API" begin
            basis = QuadPairBasis(2)
            cpu_state = coherentstate(basis, 1.0 + 0.5im)
            gpu_state_f32 = gpu(cpu_state, precision=Float32)
            gpu_state_f64 = gpu(cpu_state, precision=Float64)
            @test device(gpu_state_f32) == :gpu
            @test device(gpu_state_f64) == :gpu
            @test eltype(gpu_state_f32.mean) == Float32
            @test eltype(gpu_state_f64.mean) == Float64
            cpu_back_f32 = cpu(gpu_state_f32)
            cpu_back_f64 = cpu(gpu_state_f64)
            @test device(cpu_back_f32) == :cpu
            @test device(cpu_back_f64) == :cpu
            @test isapprox(cpu_back_f32.mean, cpu_state.mean, rtol=1e-6)
            @test isapprox(cpu_back_f64.mean, cpu_state.mean, rtol=1e-12)
        end
        
        @testset "Adapt Device Functionality" begin
            basis = QuadPairBasis(1)
            cpu_state = coherentstate(basis, 1.0)
            gpu_state = gpu(cpu_state, precision=Float32)
            new_cpu_state = adapt_device(coherentstate, cpu_state, basis, 2.0)
            new_gpu_state = adapt_device(coherentstate, gpu_state, basis, 2.0)
            @test device(new_cpu_state) == :cpu
            @test device(new_gpu_state) == :gpu
        end
    end

    @testset "GPU GaussianState Basis Conversions" begin
        nmodes_list = [1, 2, 5]
        precisions = [Float32, Float64]
        
        for nmodes in nmodes_list
            for T in precisions
                @testset "$(nmodes) modes, $(T) precision" begin
                    pair_basis = QuadPairBasis(nmodes)
                    block_basis = QuadBlockBasis(nmodes)
                    cpu_coherent_pair = coherentstate(pair_basis, 1.0 + 0.5im)
                    cpu_coherent_block = coherentstate(block_basis, 1.0 + 0.5im)
                    cpu_squeezed_pair = squeezedstate(pair_basis, 0.3, π/4)
                    cpu_squeezed_block = squeezedstate(block_basis, 0.3, π/4)
                    gpu_coherent_pair = gpu(cpu_coherent_pair, precision=T)
                    gpu_coherent_block = gpu(cpu_coherent_block, precision=T)
                    gpu_squeezed_pair = gpu(cpu_squeezed_pair, precision=T)
                    gpu_squeezed_block = gpu(cpu_squeezed_block, precision=T)
                    
                    @testset "QuadPairBasis -> QuadBlockBasis" begin
                        gpu_converted = changebasis(QuadBlockBasis, gpu_coherent_pair)
                        cpu_converted = changebasis(QuadBlockBasis, cpu_coherent_pair)
                        @test gpu_converted.basis isa QuadBlockBasis
                        @test gpu_converted.basis.nmodes == nmodes
                        @test device(gpu_converted) == :gpu
                        @test eltype(gpu_converted.mean) == T
                        @test eltype(gpu_converted.covar) == T
                        @test gpu_converted.ħ == cpu_coherent_pair.ħ
                        @test isapprox(Array(gpu_converted.mean), cpu_converted.mean, atol=1e-6)
                        @test isapprox(Array(gpu_converted.covar), cpu_converted.covar, atol=1e-6)
                        gpu_squeezed_converted = changebasis(QuadBlockBasis, gpu_squeezed_pair)
                        cpu_squeezed_converted = changebasis(QuadBlockBasis, cpu_squeezed_pair)
                        @test device(gpu_squeezed_converted) == :gpu
                        @test isapprox(Array(gpu_squeezed_converted.mean), cpu_squeezed_converted.mean, atol=1e-6)
                        @test isapprox(Array(gpu_squeezed_converted.covar), cpu_squeezed_converted.covar, atol=1e-6)
                    end
                    
                    @testset "QuadBlockBasis -> QuadPairBasis" begin
                        gpu_converted = changebasis(QuadPairBasis, gpu_coherent_block)
                        cpu_converted = changebasis(QuadPairBasis, cpu_coherent_block)
                        @test gpu_converted.basis isa QuadPairBasis
                        @test gpu_converted.basis.nmodes == nmodes
                        @test device(gpu_converted) == :gpu
                        @test eltype(gpu_converted.mean) == T
                        @test eltype(gpu_converted.covar) == T
                        @test isapprox(Array(gpu_converted.mean), cpu_converted.mean, atol=1e-6)
                        @test isapprox(Array(gpu_converted.covar), cpu_converted.covar, atol=1e-6)
                        gpu_squeezed_converted = changebasis(QuadPairBasis, gpu_squeezed_block)
                        cpu_squeezed_converted = changebasis(QuadPairBasis, cpu_squeezed_block)
                        @test device(gpu_squeezed_converted) == :gpu
                        @test isapprox(Array(gpu_squeezed_converted.mean), cpu_squeezed_converted.mean, atol=1e-6)
                        @test isapprox(Array(gpu_squeezed_converted.covar), cpu_squeezed_converted.covar, atol=1e-6)
                    end
                    
                    @testset "Same basis no-op conversions" begin
                        same_basis_pair = changebasis(QuadPairBasis, gpu_coherent_pair)
                        same_basis_block = changebasis(QuadBlockBasis, gpu_coherent_block)
                        @test device(same_basis_pair) == :gpu
                        @test device(same_basis_block) == :gpu
                        @test same_basis_pair.basis isa QuadPairBasis
                        @test same_basis_block.basis isa QuadBlockBasis
                        @test isapprox(Array(same_basis_pair.mean), Array(gpu_coherent_pair.mean), atol=1e-12)
                        @test isapprox(Array(same_basis_block.mean), Array(gpu_coherent_block.mean), atol=1e-12)
                    end
                    
                    @testset "Round-trip conversion consistency" begin
                        gpu_pair_to_block = changebasis(QuadBlockBasis, gpu_coherent_pair)
                        gpu_round_trip = changebasis(QuadPairBasis, gpu_pair_to_block)
                        @test device(gpu_round_trip) == :gpu
                        @test isapprox(Array(gpu_round_trip.mean), Array(gpu_coherent_pair.mean), atol=1e-6)
                        @test isapprox(Array(gpu_round_trip.covar), Array(gpu_coherent_pair.covar), atol=1e-6)
                        gpu_block_to_pair = changebasis(QuadPairBasis, gpu_coherent_block)
                        gpu_round_trip2 = changebasis(QuadBlockBasis, gpu_block_to_pair)
                        @test device(gpu_round_trip2) == :gpu
                        @test isapprox(Array(gpu_round_trip2.mean), Array(gpu_coherent_block.mean), atol=1e-6)
                        @test isapprox(Array(gpu_round_trip2.covar), Array(gpu_coherent_block.covar), atol=1e-6)
                    end
                end
            end
        end
    end

    @testset "GPU GaussianUnitary Basis Conversions" begin
        for nmodes in [1, 2]
            for T in [Float32, Float64]
                @testset "$(nmodes) modes, $(T) precision" begin
                    pair_basis = QuadPairBasis(nmodes)
                    block_basis = QuadBlockBasis(nmodes)
                    cpu_displace_pair = displace(pair_basis, 0.5 + 0.3im)
                    cpu_displace_block = displace(block_basis, 0.5 + 0.3im)
                    cpu_squeeze_pair = squeeze(pair_basis, 0.2, π/3)
                    cpu_squeeze_block = squeeze(block_basis, 0.2, π/3)
                    gpu_displace_pair = gpu(cpu_displace_pair, precision=T)
                    gpu_displace_block = gpu(cpu_displace_block, precision=T)
                    gpu_squeeze_pair = gpu(cpu_squeeze_pair, precision=T)
                    gpu_squeeze_block = gpu(cpu_squeeze_block, precision=T)
                    
                    @testset "Unitary basis conversions" begin
                        gpu_converted = changebasis(QuadBlockBasis, gpu_displace_pair)
                        cpu_converted = changebasis(QuadBlockBasis, cpu_displace_pair)
                        @test gpu_converted.basis isa QuadBlockBasis
                        @test device(gpu_converted) == :gpu
                        @test isapprox(Array(gpu_converted.disp), cpu_converted.disp, atol=1e-6)
                        @test isapprox(Array(gpu_converted.symplectic), cpu_converted.symplectic, atol=1e-6)
                        gpu_converted2 = changebasis(QuadPairBasis, gpu_displace_block)
                        cpu_converted2 = changebasis(QuadPairBasis, cpu_displace_block)
                        @test gpu_converted2.basis isa QuadPairBasis
                        @test device(gpu_converted2) == :gpu
                        @test isapprox(Array(gpu_converted2.disp), cpu_converted2.disp, atol=1e-6)
                        @test isapprox(Array(gpu_converted2.symplectic), cpu_converted2.symplectic, atol=1e-6)
                        gpu_squeeze_converted = changebasis(QuadBlockBasis, gpu_squeeze_pair)
                        cpu_squeeze_converted = changebasis(QuadBlockBasis, cpu_squeeze_pair)
                        @test device(gpu_squeeze_converted) == :gpu
                        @test isapprox(Array(gpu_squeeze_converted.disp), cpu_squeeze_converted.disp, atol=1e-6)
                        @test isapprox(Array(gpu_squeeze_converted.symplectic), cpu_squeeze_converted.symplectic, atol=1e-6)
                    end
                    
                    @testset "Unitary no-op conversions" begin
                        same_basis_pair = changebasis(QuadPairBasis, gpu_displace_pair)
                        same_basis_block = changebasis(QuadBlockBasis, gpu_displace_block)
                        @test device(same_basis_pair) == :gpu
                        @test device(same_basis_block) == :gpu
                        @test same_basis_pair.basis isa QuadPairBasis
                        @test same_basis_block.basis isa QuadBlockBasis
                    end
                end
            end
        end
    end

    @testset "GPU GaussianChannel Basis Conversions" begin
        for nmodes in [1, 2]
            for T in [Float32, Float64]
                @testset "$(nmodes) modes, $(T) precision" begin
                    pair_basis = QuadPairBasis(nmodes)
                    block_basis = QuadBlockBasis(nmodes)
                    cpu_attenuator_pair = attenuator(pair_basis, π/4, 2)
                    cpu_attenuator_block = attenuator(block_basis, π/4, 2)
                    cpu_amplifier_pair = amplifier(pair_basis, 0.3, 3)
                    cpu_amplifier_block = amplifier(block_basis, 0.3, 3)
                    gpu_attenuator_pair = gpu(cpu_attenuator_pair, precision=T)
                    gpu_attenuator_block = gpu(cpu_attenuator_block, precision=T)
                    gpu_amplifier_pair = gpu(cpu_amplifier_pair, precision=T)
                    gpu_amplifier_block = gpu(cpu_amplifier_block, precision=T)
                    
                    @testset "Channel basis conversions" begin
                        gpu_converted = changebasis(QuadBlockBasis, gpu_attenuator_pair)
                        cpu_converted = changebasis(QuadBlockBasis, cpu_attenuator_pair)
                        @test gpu_converted.basis isa QuadBlockBasis
                        @test device(gpu_converted) == :gpu
                        @test isapprox(Array(gpu_converted.disp), cpu_converted.disp, atol=1e-6)
                        @test isapprox(Array(gpu_converted.transform), cpu_converted.transform, atol=1e-6)
                        @test isapprox(Array(gpu_converted.noise), cpu_converted.noise, atol=1e-6)
                        gpu_converted2 = changebasis(QuadPairBasis, gpu_attenuator_block)
                        cpu_converted2 = changebasis(QuadPairBasis, cpu_attenuator_block)
                        @test gpu_converted2.basis isa QuadPairBasis
                        @test device(gpu_converted2) == :gpu
                        @test isapprox(Array(gpu_converted2.disp), cpu_converted2.disp, atol=1e-6)
                        @test isapprox(Array(gpu_converted2.transform), cpu_converted2.transform, atol=1e-6)
                        @test isapprox(Array(gpu_converted2.noise), cpu_converted2.noise, atol=1e-6)
                        gpu_amp_converted = changebasis(QuadBlockBasis, gpu_amplifier_pair)
                        cpu_amp_converted = changebasis(QuadBlockBasis, cpu_amplifier_pair)
                        @test device(gpu_amp_converted) == :gpu
                        @test isapprox(Array(gpu_amp_converted.disp), cpu_amp_converted.disp, atol=1e-6)
                        @test isapprox(Array(gpu_amp_converted.transform), cpu_amp_converted.transform, atol=1e-6)
                        @test isapprox(Array(gpu_amp_converted.noise), cpu_amp_converted.noise, atol=1e-6)
                    end
                    
                    @testset "Channel no-op conversions" begin
                        same_basis_pair = changebasis(QuadPairBasis, gpu_attenuator_pair)
                        same_basis_block = changebasis(QuadBlockBasis, gpu_attenuator_block)
                        @test device(same_basis_pair) == :gpu
                        @test device(same_basis_block) == :gpu
                        @test same_basis_pair.basis isa QuadPairBasis
                        @test same_basis_block.basis isa QuadBlockBasis
                    end
                end
            end
        end
    end

    @testset "CPU/GPU Mixed Operations with Basis Conversion" begin
        basis_pair = QuadPairBasis(2)
        basis_block = QuadBlockBasis(2)
        cpu_state_pair = coherentstate(basis_pair, 1.0 + 0.5im)
        gpu_state_pair = gpu(coherentstate(basis_pair, 0.5 - 0.3im), precision=Float32)
        cpu_state_block = coherentstate(basis_block, 2.0 + 0.1im)
        gpu_state_block = gpu(coherentstate(basis_block, -0.8 + 0.7im), precision=Float32)
        
        @testset "Mixed device basis conversions" begin
            cpu_pair_to_block = changebasis(QuadBlockBasis, cpu_state_pair)
            @test device(cpu_pair_to_block) == :cpu
            @test cpu_pair_to_block.basis isa QuadBlockBasis
            gpu_pair_to_block = changebasis(QuadBlockBasis, gpu_state_pair)
            @test device(gpu_pair_to_block) == :gpu
            @test gpu_pair_to_block.basis isa QuadBlockBasis
            cpu_to_gpu_converted = gpu(changebasis(QuadBlockBasis, cpu_state_pair), precision=Float32)
            @test device(cpu_to_gpu_converted) == :gpu
            @test cpu_to_gpu_converted.basis isa QuadBlockBasis
            gpu_to_cpu_converted = cpu(changebasis(QuadPairBasis, gpu_state_block))
            @test device(gpu_to_cpu_converted) == :cpu
            @test gpu_to_cpu_converted.basis isa QuadPairBasis
        end
        
        @testset "Operations across converted states" begin
            cpu_op_pair = displace(basis_pair, 0.3)
            gpu_op_block = gpu(displace(basis_block, 0.4), precision=Float32)
            converted_state = changebasis(QuadBlockBasis, gpu_state_pair)
            result = gpu_op_block * converted_state
            @test device(result) == :gpu
            @test result.basis isa QuadBlockBasis
            cpu_state_converted = changebasis(QuadPairBasis, cpu_state_block)
            result2 = cpu_op_pair * cpu_state_converted
            @test device(result2) == :cpu
            @test result2.basis isa QuadPairBasis
        end
    end

    @testset "GPU Basis Conversion Error Handling" begin
        basis_pair = QuadPairBasis(2)
        cpu_state = coherentstate(basis_pair, 1.0 + 0.5im)
        converted_cpu = changebasis(QuadBlockBasis, cpu_state)
        @test converted_cpu.basis isa QuadBlockBasis
        @test device(converted_cpu) == :cpu
        @test isgaussian(converted_cpu, atol=1e-4)
        squeezed_cpu = squeezedstate(basis_pair, 0.5, π/4)
        converted_squeezed = changebasis(QuadBlockBasis, squeezed_cpu)
        @test isgaussian(converted_squeezed, atol=1e-4)
        thermal_cpu = thermalstate(basis_pair, 2.0)
        converted_thermal = changebasis(QuadBlockBasis, thermal_cpu)
        @test isgaussian(converted_thermal, atol=1e-4)
    end

    @testset "Memory Efficiency and Performance Validation" begin
        basis_pair = QuadPairBasis(10)
        large_state = gpu(squeezedstate(basis_pair, 0.5, π/6), precision=Float32)
        @test device(large_state) == :gpu
        @test eltype(large_state.mean) == Float32
        converted_state = changebasis(QuadBlockBasis, large_state)
        @test device(converted_state) == :gpu
        @test converted_state.basis isa QuadBlockBasis
        @test eltype(converted_state.mean) == Float32
        @test eltype(converted_state.covar) == Float32
        round_trip = changebasis(QuadPairBasis, converted_state)
        @test device(round_trip) == :gpu
        @test isapprox(Array(round_trip.mean), Array(large_state.mean), atol=1e-6)
        @test isapprox(Array(round_trip.covar), Array(large_state.covar), atol=1e-6)
    end

    @testset "Integration with Linear Combinations" begin
        basis_pair = QuadPairBasis(2)
        state1 = gpu(coherentstate(basis_pair, 1.0), precision=Float32)
        state2 = gpu(coherentstate(basis_pair, -1.0), precision=Float32)
        lc = GaussianLinearCombination(basis_pair, [0.7f0, 0.3f0], [state1, state2])
        @test device(lc) == :gpu
        @test lc.basis isa QuadPairBasis
        converted_state1 = changebasis(QuadBlockBasis, state1)
        converted_state2 = changebasis(QuadBlockBasis, state2)
        @test device(converted_state1) == :gpu
        @test device(converted_state2) == :gpu
        @test converted_state1.basis isa QuadBlockBasis
        @test converted_state2.basis isa QuadBlockBasis
        lc_converted = GaussianLinearCombination(QuadBlockBasis(2), [0.7f0, 0.3f0], 
                                                [converted_state1, converted_state2])
        @test device(lc_converted) == :gpu
        @test lc_converted.basis isa QuadBlockBasis
    end
end

@testitem "GPU Random Generation Suite" begin
    using Gabs
    using Gabs: device, GaussianState, GaussianUnitary, GaussianChannel
    using CUDA
    using LinearAlgebra: adjoint, inv, eigvals, det
    using Test

    if !CUDA.functional()
        @warn "CUDA not functional. Skipping GPU random generation tests."
        return
    end

    @testset "GPU Random States - Physical Properties" begin
        nmodes = rand(1:3)
        qpairbasis = QuadPairBasis(nmodes)
        qblockbasis = QuadBlockBasis(nmodes)
        
        @testset "Pure Random States - Convenience API" begin
            rs_pair_gpu = randstate_gpu(qpairbasis; pure=true)
            rs_block_gpu = randstate_gpu(qblockbasis; pure=true)
            @test rs_pair_gpu isa GaussianState
            @test rs_block_gpu isa GaussianState
            @test device(rs_pair_gpu) == :gpu
            @test device(rs_block_gpu) == :gpu
            @test rs_pair_gpu.ħ == 2
            @test rs_block_gpu.ħ == 2
            rs_pair_cpu = cpu(rs_pair_gpu)
            rs_block_cpu = cpu(rs_block_gpu)
            @test isgaussian(rs_pair_cpu, atol = 1e-3)
            @test isgaussian(rs_block_cpu, atol = 1e-3)
            @test isapprox(purity(rs_pair_cpu), 1.0, atol = 1e-3)
            @test isapprox(purity(rs_block_cpu), 1.0, atol = 1e-3)
        end
        
        @testset "Mixed Random States - Convenience API" begin
            rs_mixed_gpu = randstate_gpu(qpairbasis; pure=false)
            @test rs_mixed_gpu isa GaussianState
            @test device(rs_mixed_gpu) == :gpu
            rs_mixed_cpu = cpu(rs_mixed_gpu)
            @test isgaussian(rs_mixed_cpu, atol = 1e-3)
        end
        
        @testset "Typed Random States - Your GPU Extensions" begin
            rs_typed_gpu = randstate(CuVector{Float32}, CuMatrix{Float32}, qpairbasis; pure=true)
            @test rs_typed_gpu isa GaussianState
            @test device(rs_typed_gpu) == :gpu
            @test eltype(rs_typed_gpu.mean) == Float32
            @test eltype(rs_typed_gpu.covar) == Float32
            rs_typed_cpu = cpu(rs_typed_gpu)
            @test isgaussian(rs_typed_cpu, atol = 1e-3)
            @test isapprox(purity(rs_typed_cpu), 1.0, atol = 1e-3)
        end
        
        @testset "Batch States - Your Batch Functions" begin
            batch_size = 3
            states_gpu = batch_randstate_gpu(qpairbasis, batch_size; pure=true)
            @test length(states_gpu) == batch_size
            @test all(s isa GaussianState for s in states_gpu)
            @test all(device(s) == :gpu for s in states_gpu)
            state_cpu = cpu(states_gpu[1])
            @test isgaussian(state_cpu, atol = 1e-3)
            @test isapprox(purity(state_cpu), 1.0, atol = 1e-3)
        end
    end

    @testset "GPU Random Unitaries - Your GPU Functions" begin
        nmodes = rand(1:3)
        qpairbasis = QuadPairBasis(nmodes)
        
        @testset "Passive Unitaries - Convenience API" begin
            ru_passive_gpu = randunitary_gpu(qpairbasis; passive=true)
            @test ru_passive_gpu isa GaussianUnitary
            @test device(ru_passive_gpu) == :gpu
            @test ru_passive_gpu.ħ == 2
            S_cpu = Array(ru_passive_gpu.symplectic)
            @test isapprox(S_cpu', inv(S_cpu), atol = 1e-3)
            @test issymplectic(qpairbasis, S_cpu, atol = 1e-3)
        end
        
        @testset "Active Unitaries - Convenience API" begin
            ru_active_gpu = randunitary_gpu(qpairbasis; passive=false)
            @test ru_active_gpu isa GaussianUnitary
            @test device(ru_active_gpu) == :gpu
            @test ru_active_gpu.ħ == 2
            S_cpu = Array(ru_active_gpu.symplectic)
            @test issymplectic(qpairbasis, S_cpu, atol = 1e-3)
        end
        
        @testset "Typed Unitaries - Your GPU Extensions" begin
            ru_typed_gpu = randunitary(CuVector{Float32}, CuMatrix{Float32}, qpairbasis; passive=false)
            @test ru_typed_gpu isa GaussianUnitary
            @test device(ru_typed_gpu) == :gpu
            @test eltype(ru_typed_gpu.disp) == Float32
            @test eltype(ru_typed_gpu.symplectic) == Float32
            S_cpu = Array(ru_typed_gpu.symplectic)
            @test issymplectic(qpairbasis, S_cpu, atol = 1e-3)
        end
        
        @testset "Batch Unitaries - Your Batch Functions" begin
            batch_size = 3
            unitaries_gpu = batch_randunitary_gpu(qpairbasis, batch_size; passive=false)
            @test length(unitaries_gpu) == batch_size
            @test all(u isa GaussianUnitary for u in unitaries_gpu)
            @test all(device(u) == :gpu for u in unitaries_gpu)
            S_cpu = Array(unitaries_gpu[1].symplectic)
            @test issymplectic(qpairbasis, S_cpu, atol = 1e-3)
        end
    end

    @testset "GPU Random Channels - Your GPU Functions" begin
        nmodes = rand(1:2)
        qpairbasis = QuadPairBasis(nmodes)
        
        @testset "Basic Channel Generation - Convenience API" begin
            rc_gpu = randchannel_gpu(qpairbasis)
            @test rc_gpu isa GaussianChannel
            @test device(rc_gpu) == :gpu
            @test rc_gpu.ħ == 2
            rc_cpu = cpu(rc_gpu)
            @test isgaussian(rc_cpu, atol = 1e-3)
            state_gpu = randstate_gpu(qpairbasis)
            output_gpu = rc_gpu * state_gpu
            @test output_gpu isa GaussianState
            @test device(output_gpu) == :gpu
        end
        
        @testset "Typed Channels - Your GPU Extensions" begin
            rc_typed_gpu = randchannel(CuVector{Float32}, CuMatrix{Float32}, qpairbasis)
            @test rc_typed_gpu isa GaussianChannel
            @test device(rc_typed_gpu) == :gpu
            @test eltype(rc_typed_gpu.disp) == Float32
            @test eltype(rc_typed_gpu.transform) == Float32
            @test eltype(rc_typed_gpu.noise) == Float32
            rc_cpu = cpu(rc_typed_gpu)
            @test isgaussian(rc_cpu, atol = 1e-3)
        end
    end

    @testset "GPU Random Symplectic Matrices - Your GPU Functions" begin
        nmodes = rand(1:3)
        qpairbasis = QuadPairBasis(nmodes)
        qblockbasis = QuadBlockBasis(nmodes)
        
        @testset "Passive Symplectic - Convenience API" begin
            S_passive_gpu = randsymplectic_gpu(qpairbasis; passive=true)
            @test S_passive_gpu isa CuMatrix
            @test size(S_passive_gpu) == (2*nmodes, 2*nmodes)
            S_passive_cpu = Array(S_passive_gpu)
            @test isapprox(S_passive_cpu', inv(S_passive_cpu), atol = 1e-3)
            @test issymplectic(qpairbasis, S_passive_cpu, atol = 1e-3)
        end
        
        @testset "Active Symplectic - Convenience API" begin
            S_active_gpu = randsymplectic_gpu(qblockbasis; passive=false)
            @test S_active_gpu isa CuMatrix
            @test size(S_active_gpu) == (2*nmodes, 2*nmodes)
            S_active_cpu = Array(S_active_gpu)
            @test issymplectic(qblockbasis, S_active_cpu, atol = 1e-3)
        end
        
        @testset "Typed Symplectic - Your GPU Extensions" begin
            S_typed_gpu = randsymplectic(CuMatrix{Float32}, qpairbasis; passive=true)
            @test S_typed_gpu isa CuMatrix{Float32}
            @test eltype(S_typed_gpu) == Float32
            @test size(S_typed_gpu) == (2*nmodes, 2*nmodes)
            S_typed_cpu = Array(S_typed_gpu)
            @test isapprox(S_typed_cpu', inv(S_typed_cpu), atol = 1e-3)
            @test issymplectic(qpairbasis, S_typed_cpu, atol = 1e-3)
        end
    end

    @testset "GPU Random Integration - Your Functions Working Together" begin
        nmodes = 2
        qpairbasis = QuadPairBasis(nmodes)
        
        @testset "State-Unitary Operations" begin
            state_gpu = randstate_gpu(qpairbasis; pure=true)
            unitary_gpu = randunitary_gpu(qpairbasis; passive=false)
            transformed_gpu = unitary_gpu * state_gpu
            @test transformed_gpu isa GaussianState
            @test device(transformed_gpu) == :gpu
            transformed_cpu = cpu(transformed_gpu)
            @test isgaussian(transformed_cpu, atol = 1e-3)
        end
        
        @testset "State-Channel Operations" begin
            state_gpu = randstate_gpu(qpairbasis; pure=false)
            channel_gpu = randchannel_gpu(qpairbasis)
            output_gpu = channel_gpu * state_gpu
            @test output_gpu isa GaussianState
            @test device(output_gpu) == :gpu
            output_cpu = cpu(output_gpu)
            @test isgaussian(output_cpu, atol = 1e-3)
        end
    end

    @testset "GPU Random Error Handling - Your API" begin
        qpairbasis = QuadPairBasis(1)
        @test_nowarn randstate(CuVector{Float32}, CuMatrix{Float32}, qpairbasis)
        @test_nowarn randunitary(CuVector{Float32}, CuMatrix{Float32}, qpairbasis)
        @test_nowarn randchannel(CuVector{Float32}, CuMatrix{Float32}, qpairbasis)
        @test_nowarn randsymplectic(CuMatrix{Float32}, qpairbasis)
        @test_nowarn randstate_gpu(qpairbasis)
        @test_nowarn randunitary_gpu(qpairbasis)
        @test_nowarn randchannel_gpu(qpairbasis)
        @test_nowarn randsymplectic_gpu(qpairbasis)
        @test_nowarn batch_randstate_gpu(qpairbasis, 2)
        @test_nowarn batch_randunitary_gpu(qpairbasis, 2)
    end

    @testset "GPU Random Precision Control - Your Features" begin
        qpairbasis = QuadPairBasis(1)
        
        @testset "Float32 Precision" begin
            state_f32 = randstate_gpu(qpairbasis; precision=Float32)
            unitary_f32 = randunitary_gpu(qpairbasis; precision=Float32)
            symp_f32 = randsymplectic_gpu(qpairbasis; precision=Float32)
            @test eltype(state_f32.mean) == Float32
            @test eltype(unitary_f32.disp) == Float32
            @test eltype(symp_f32) == Float32
        end
        
        @testset "Float64 Precision" begin
            state_f64 = randstate_gpu(qpairbasis; precision=Float64)
            unitary_f64 = randunitary_gpu(qpairbasis; precision=Float64)
            symp_f64 = randsymplectic_gpu(qpairbasis; precision=Float64)
            @test eltype(state_f64.mean) == Float64
            @test eltype(unitary_f64.disp) == Float64
            @test eltype(symp_f64) == Float64
        end
    end

    @testset "GPU Random Performance - Your Batch Functions" begin
        qpairbasis = QuadPairBasis(2)
        
        @testset "Batch vs Individual Performance" begin
            batch_size = 5
            batch_states = batch_randstate_gpu(qpairbasis, batch_size; pure=true)
            batch_unitaries = batch_randunitary_gpu(qpairbasis, batch_size; passive=false)
            @test length(batch_states) == batch_size
            @test length(batch_unitaries) == batch_size
            @test all(device(s) == :gpu for s in batch_states)
            @test all(device(u) == :gpu for u in batch_unitaries)
            state_cpu = cpu(batch_states[1])
            unitary_cpu = cpu(batch_unitaries[1])
            @test isgaussian(state_cpu, atol = 1e-3)
            @test issymplectic(qpairbasis, Array(batch_unitaries[1].symplectic), atol = 1e-3)
        end
    end
end

@testitem "GPU Cross-Wigner Foundation - single thread, but accuracy tests" begin
    using Gabs
    using Gabs: device, cross_wigner, cross_wignerchar
    using LinearAlgebra
    using CUDA

    @testset "GPU Cross-Wigner Function Correctness" begin
        basis = QuadPairBasis(1)
        state1_cpu = coherentstate(basis, 1.0 + 0.5im)
        state2_cpu = coherentstate(basis, -0.8 + 0.3im)
        state1_gpu = GaussianState(basis, CuArray{Float32}(state1_cpu.mean), CuArray{Float32}(state1_cpu.covar); ħ = state1_cpu.ħ)
        state2_gpu = GaussianState(basis, CuArray{Float32}(state2_cpu.mean), CuArray{Float32}(state2_cpu.covar); ħ = state2_cpu.ħ)
        test_points = [
            [0.0, 0.0],
            [1.0, 0.5],
            [-0.5, 1.2],
            [0.3, -0.8],
            [2.1, -1.4]
        ]
        
        @testset "Cross-Wigner vs CPU Implementation" begin
            for x in test_points
                cpu_result = cross_wigner(state1_cpu, state2_cpu, x)
                gpu_result = cross_wigner(state1_gpu, state2_gpu, x)
                @test real(cpu_result) ≈ real(gpu_result) atol=1e-6 rtol=1e-5
                @test imag(cpu_result) ≈ imag(gpu_result) atol=1e-6 rtol=1e-5
            end
        end
        
        @testset "Cross-Wigner Hermiticity Property" begin
            for x in test_points
                w12 = cross_wigner(state1_gpu, state2_gpu, x)
                w21 = cross_wigner(state2_gpu, state1_gpu, x)
                @test real(w12) ≈ real(w21) atol=1e-6 rtol=1e-5
                @test imag(w12) ≈ -imag(w21) atol=1e-6 rtol=1e-5
                @test w12 ≈ conj(w21) atol=1e-6 rtol=1e-5
            end
        end
        
        @testset "Cross-Wigner Identity Property" begin  
            for x in test_points
                cross_diag = cross_wigner(state1_gpu, state1_gpu, x)
                regular_wigner = wigner(state1_gpu, x)
                @test real(cross_diag) ≈ regular_wigner atol=1e-6 rtol=1e-5
                @test abs(imag(cross_diag)) < 1e-6
            end
        end
    end

    @testset "GPU Cross-Wigner Characteristic Function Correctness" begin
        basis = QuadPairBasis(1)
        state1_cpu = squeezedstate(basis, 0.3, π/4)
        state2_cpu = coherentstate(basis, 0.5 - 0.2im)
        state1_gpu = GaussianState(basis, CuArray{Float32}(state1_cpu.mean), CuArray{Float32}(state1_cpu.covar); ħ = state1_cpu.ħ)
        state2_gpu = GaussianState(basis, CuArray{Float32}(state2_cpu.mean), CuArray{Float32}(state2_cpu.covar); ħ = state2_cpu.ħ)
        test_xi = [
            [0.0, 0.0], 
            [0.1, -0.3],
            [-0.7, 0.4],
            [1.2, 0.8],
            [-0.9, -1.1]
        ]
        
        @testset "Cross-WignerChar vs CPU Implementation" begin
            for xi in test_xi
                cpu_result = cross_wignerchar(state1_cpu, state2_cpu, xi)
                gpu_result = cross_wignerchar(state1_gpu, state2_gpu, xi)
                @test real(cpu_result) ≈ real(gpu_result) atol=1e-6 rtol=1e-5
                @test imag(cpu_result) ≈ imag(gpu_result) atol=1e-6 rtol=1e-5
            end
        end
        
        @testset "Cross-WignerChar Hermiticity Property" begin
            for xi in test_xi
                chi12 = cross_wignerchar(state1_gpu, state2_gpu, xi)
                chi21 = cross_wignerchar(state2_gpu, state1_gpu, -xi)
                @test real(chi12) ≈ real(chi21) atol=1e-6 rtol=1e-5
                @test imag(chi12) ≈ -imag(chi21) atol=1e-6 rtol=1e-5
            end
        end
        
        @testset "Cross-WignerChar Identity Property" begin
            for xi in test_xi
                cross_diag = cross_wignerchar(state1_gpu, state1_gpu, xi)
                regular_char = wignerchar(state1_gpu, xi)
                @test abs(cross_diag - regular_char) < 1e-6
            end
        end
    end

    @testset "GPU Cross-Wigner Multi-mode Systems" begin
        basis = QuadPairBasis(2)
        state1_cpu = coherentstate(basis, [1.0 + 0.5im, -0.3 + 0.8im])
        state2_cpu = squeezedstate(basis, [0.2, 0.4], [π/6, π/3])
        state1_gpu = GaussianState(basis, CuArray{Float32}(state1_cpu.mean), CuArray{Float32}(state1_cpu.covar); ħ = state1_cpu.ħ)
        state2_gpu = GaussianState(basis, CuArray{Float32}(state2_cpu.mean), CuArray{Float32}(state2_cpu.covar); ħ = state2_cpu.ħ)
        test_points_2mode = [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.5, -0.3, 0.8],
            [-0.2, 1.1, 0.7, -0.4]
        ]
        
        @testset "Multi-mode Cross-Wigner Correctness" begin
            for x in test_points_2mode
                cpu_result = cross_wigner(state1_cpu, state2_cpu, x)
                gpu_result = cross_wigner(state1_gpu, state2_gpu, x)
                @test real(cpu_result) ≈ real(gpu_result) atol=1e-5 rtol=1e-4
                @test imag(cpu_result) ≈ imag(gpu_result) atol=1e-5 rtol=1e-4
            end
        end
        
        @testset "Multi-mode Cross-WignerChar Correctness" begin
            for xi in test_points_2mode
                cpu_result = cross_wignerchar(state1_cpu, state2_cpu, xi)
                gpu_result = cross_wignerchar(state1_gpu, state2_gpu, xi)
                @test real(cpu_result) ≈ real(gpu_result) atol=1e-5 rtol=1e-4
                @test imag(cpu_result) ≈ imag(gpu_result) atol=1e-5 rtol=1e-4
            end
        end
    end

    @testset "GPU Cross-Wigner Mixed Device Support" begin
        basis = QuadPairBasis(1)
        state1_gpu = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, 0.8 + 0.6im)
        state2_gpu = squeezedstate(CuVector{Float32}, CuMatrix{Float32}, basis, 0.4, π/8)
        x_cpu = [0.5, -0.3]
        xi_cpu = [0.2, 0.8]
        
        @testset "GPU States with CPU Points" begin
            result_w = cross_wigner(state1_gpu, state2_gpu, x_cpu)
            result_chi = cross_wignerchar(state1_gpu, state2_gpu, xi_cpu)
            @test result_w isa ComplexF32
            @test result_chi isa ComplexF32
            @test isfinite(real(result_w)) && isfinite(imag(result_w))
            @test isfinite(real(result_chi)) && isfinite(imag(result_chi))
        end
    end

    @testset "GPU Cross-Wigner Error Handling" begin
        basis1 = QuadPairBasis(1)
        basis2 = QuadBlockBasis(1)
        state1 = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis1, 1.0)
        state2 = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis2, 1.0)
        state3 = GaussianState(basis1, CuArray{Float32}([1.0, 0.0]), CuArray{Float32}([1.0 0.0; 0.0 1.0]); ħ = 4)
        x = [0.0, 0.0]
        
        @testset "Basis Mismatch Detection" begin
            @test_throws ArgumentError cross_wigner(state1, state2, x)
            @test_throws ArgumentError cross_wignerchar(state1, state2, x)
        end
        
        @testset "ħ Mismatch Detection" begin
            @test_throws ArgumentError cross_wigner(state1, state3, x)
            @test_throws ArgumentError cross_wignerchar(state1, state3, x)
        end
        
        @testset "Dimension Mismatch Detection" begin
            wrong_x = [0.0]
            @test_throws ArgumentError cross_wigner(state1, state1, wrong_x)
            @test_throws ArgumentError cross_wignerchar(state1, state1, wrong_x)
        end
    end

    @testset "GPU Cross-Wigner Performance Validation" begin
        basis = QuadPairBasis(2)
        state1_gpu = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, [1.0 + 0.5im, -0.3 + 0.8im])
        state2_gpu = eprstate(CuVector{Float32}, CuMatrix{Float32}, basis, 0.3, π/4)
        x = [0.1, 0.2, 0.3, 0.4]
        xi = [0.5, 0.6, 0.7, 0.8]
        
        @testset "GPU Functions Execute Successfully" begin
            @test begin
                result_w = cross_wigner(state1_gpu, state2_gpu, x)
                result_chi = cross_wignerchar(state1_gpu, state2_gpu, xi)
                isfinite(real(result_w)) && isfinite(imag(result_w)) &&
                isfinite(real(result_chi)) && isfinite(imag(result_chi))
            end
        end
    end
end

@testitem "GPU Full Interference Wigner Functions" begin
    using Gabs
    using Gabs: device
    using LinearAlgebra
    using CUDA
    
    if CUDA.functional()
        @testset "Single-State Linear Combinations" begin
            basis = QuadPairBasis(1)
            alpha = 1.0f0 + 0.5f0*im
            gpu_state = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, alpha)
            coeffs = [1.0f0]
            lc_single = GaussianLinearCombination(basis, coeffs, [gpu_state])
            x_cpu = [0.1f0, 0.3f0]
            xi_cpu = [0.2f0, -0.4f0]
            x_gpu = CuArray(x_cpu)
            xi_gpu = CuArray(xi_cpu)
            w_single_direct = wigner(gpu_state, x_gpu)
            w_single_interference = wigner(lc_single, x_gpu)
            @test w_single_direct ≈ w_single_interference rtol=1e-6
            char_single_direct = wignerchar(gpu_state, xi_gpu)
            char_single_interference = wignerchar(lc_single, xi_gpu)
            @test char_single_direct ≈ char_single_interference rtol=1e-6
            w_mixed = wigner(lc_single, x_cpu)
            @test w_single_direct ≈ w_mixed rtol=1e-6
            char_mixed = wignerchar(lc_single, xi_cpu)
            @test char_single_direct ≈ char_mixed rtol=1e-6
        end
        
        @testset "Two-State Interference" begin
            basis = QuadPairBasis(1)
            alpha = 1.0f0
            state_plus = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, alpha)
            state_minus = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, -alpha)
            norm_factor = 1.0f0 / sqrt(2.0f0 * (1.0f0 + exp(-2.0f0 * abs2(alpha))))
            coeffs = [norm_factor, norm_factor]
            cat_even = GaussianLinearCombination(basis, coeffs, [state_plus, state_minus])
            x_test = [0.0f0, 0.0f0]
            xi_test = [0.1f0, 0.2f0]
            w_interference = wigner(cat_even, x_test)
            char_interference = wignerchar(cat_even, xi_test)
            w_plus = wigner(state_plus, CuArray(x_test))
            w_minus = wigner(state_minus, CuArray(x_test))
            cross_w = cross_wigner(state_plus, state_minus, CuArray(x_test))
            w_manual = abs2(coeffs[1]) * w_plus + abs2(coeffs[2]) * w_minus + 
                       2 * real(conj(coeffs[1]) * coeffs[2] * cross_w)
            @test w_interference ≈ w_manual rtol=1e-6
            @test isreal(w_interference)
            char_plus = wignerchar(state_plus, CuArray(xi_test))
            char_minus = wignerchar(state_minus, CuArray(xi_test))
            cross_char = cross_wignerchar(state_plus, state_minus, CuArray(xi_test))
            char_manual = abs2(coeffs[1]) * char_plus + abs2(coeffs[2]) * char_minus + 
                          2 * real(conj(coeffs[1]) * coeffs[2] * cross_char)
            @test char_interference ≈ char_manual rtol=1e-6
        end
        
        @testset "Multi-State Interference" begin
            basis = QuadPairBasis(1)
            alphas = [1.0f0, 0.0f0, -1.0f0]
            states = [coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, α) for α in alphas]
            coeffs = [0.5f0, 0.3f0, 0.2f0]
            lc_multi = GaussianLinearCombination(basis, coeffs, states)
            x_test = [0.5f0, -0.3f0]
            xi_test = [0.2f0, 0.4f0]
            w_gpu = wigner(lc_multi, x_test)
            char_gpu = wignerchar(lc_multi, xi_test)
            w_diagonal = sum(abs2(coeffs[i]) * wigner(states[i], CuArray(x_test)) for i in 1:3)
            w_cross = 0.0f0
            for i in 1:2
                for j in (i+1):3
                    cross_term = 2 * real(conj(coeffs[i]) * coeffs[j] * 
                                         cross_wigner(states[i], states[j], CuArray(x_test)))
                    w_cross += cross_term
                end
            end
            w_manual = w_diagonal + w_cross
            @test w_gpu ≈ w_manual rtol=1e-5
            char_diagonal = sum(abs2(coeffs[i]) * wignerchar(states[i], CuArray(xi_test)) for i in 1:3)
            char_cross = complex(0.0f0)
            for i in 1:2
                for j in (i+1):3
                    cross_term = 2 * real(conj(coeffs[i]) * coeffs[j] * 
                                         cross_wignerchar(states[i], states[j], CuArray(xi_test)))
                    char_cross += cross_term
                end
            end
            char_manual = char_diagonal + char_cross
            @test char_gpu ≈ char_manual rtol=1e-5
        end
        
        @testset "Complex Coefficients" begin
            basis = QuadPairBasis(1)
            alpha1, alpha2 = 1.0f0, -0.5f0
            state1 = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, alpha1)
            state2 = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, alpha2)
            c1 = 0.6f0 + 0.2f0*im
            c2 = 0.3f0 - 0.4f0*im
            coeffs = ComplexF32[c1, c2]
            lc_complex = GaussianLinearCombination(basis, coeffs, [state1, state2])
            x_test = [0.2f0, 0.1f0]
            w_gpu = wigner(lc_complex, x_test)
            w1 = wigner(state1, CuArray(x_test))
            w2 = wigner(state2, CuArray(x_test))
            cross_w = cross_wigner(state1, state2, CuArray(x_test))
            w_manual = abs2(c1) * w1 + abs2(c2) * w2 + 2 * real(conj(c1) * c2 * cross_w)
            @test w_gpu ≈ w_manual rtol=1e-6
            @test isreal(w_gpu)
        end
        
        @testset "Multi-Mode Systems" begin
            basis = QuadPairBasis(2)
            alpha1 = ComplexF32[1.0f0, 0.5f0]
            alpha2 = ComplexF32[-0.5f0, 1.0f0]
            state1 = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, alpha1)
            state2 = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, alpha2)
            coeffs = [0.7f0, 0.3f0]
            lc_multimode = GaussianLinearCombination(basis, coeffs, [state1, state2])
            x_test = [0.1f0, 0.2f0, -0.1f0, 0.3f0]
            xi_test = [0.05f0, -0.1f0, 0.15f0, -0.05f0]
            w_gpu = wigner(lc_multimode, x_test)
            char_gpu = wignerchar(lc_multimode, xi_test)
            @test isfinite(w_gpu)
            @test isreal(w_gpu)
            @test isfinite(char_gpu)
            w1 = wigner(state1, CuArray(x_test))
            w2 = wigner(state2, CuArray(x_test))
            cross_w = cross_wigner(state1, state2, CuArray(x_test))
            w_manual = abs2(coeffs[1]) * w1 + abs2(coeffs[2]) * w2 + 
                    2 * real(conj(coeffs[1]) * coeffs[2] * cross_w)
            @test w_gpu ≈ w_manual rtol=1e-6
        end
        
        @testset "Error Handling" begin
            basis = QuadPairBasis(1)
            state = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, 1.0f0)
            lc = GaussianLinearCombination(basis, [1.0f0], [state])
            @test_throws ArgumentError wigner(lc, [0.0f0])
            @test_throws ArgumentError wignerchar(lc, [0.0f0, 0.0f0, 0.0f0])
            basis2 = QuadPairBasis(2)
            state1_2mode = tensor(state, state)
            lc2 = GaussianLinearCombination(basis2, [1.0f0], [state1_2mode])
            @test_throws ArgumentError wigner(lc2, [0.0f0, 0.0f0])
            @test_throws ArgumentError wignerchar(lc2, [0.0f0, 0.0f0, 0.0f0])
        end
        
        @testset "Performance and Memory Efficiency" begin
            basis = QuadPairBasis(1)
            n_states = 5
            states = [coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, 
                                   Float32(i) * (1.0f0 + 0.1f0*im)) for i in 1:n_states]
            coeffs = normalize([Float32(i) for i in 1:n_states])
            lc_large = GaussianLinearCombination(basis, coeffs, states)
            x_test = [0.0f0, 0.0f0]
            w_result = wigner(lc_large, x_test)
            elapsed_time = @elapsed begin
                for _ in 1:100
                    wigner(lc_large, x_test)
                end
            end
            @test elapsed_time < 1.0
            @test isfinite(w_result)
            @test isreal(w_result)
            CUDA.reclaim()
            initial_memory = CUDA.used_memory()
            w_large = wigner(lc_large, x_test)
            peak_memory = CUDA.used_memory()
            memory_increase = peak_memory - initial_memory
            @test memory_increase < 100_000_000
            @test isfinite(w_large)
        end
        
        @testset "Numerical Stability" begin
            basis = QuadPairBasis(1)
            small_alpha = 1.0f-6
            large_alpha = 1.0f6
            small_state = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, small_alpha)
            large_state = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, large_alpha)
            coeffs = [1.0f0/sqrt(2.0f0), 1.0f0/sqrt(2.0f0)]
            lc_extreme = GaussianLinearCombination(basis, coeffs, [small_state, large_state])
            x_test = [0.0f0, 0.0f0]
            w_extreme = wigner(lc_extreme, x_test)
            char_extreme = wignerchar(lc_extreme, [0.1f0, 0.1f0])
            @test isfinite(w_extreme)
            @test isfinite(char_extreme)
            @test !isnan(w_extreme)
            @test !isnan(char_extreme)
        end
    else
        @test_skip "CUDA not available, skipping GPU interference tests"
    end
end

@testitem "GPU Batched Interference Test" begin
using Gabs
using Gabs: device
using LinearAlgebra  
using CUDA
using Test

@testset "GPU Batched Interference Wigner Functions" begin
    basis = QuadPairBasis(1)
    α1, α2 = 1.0f0 + 0.5f0*im, -1.0f0 + 0.3f0*im
    state1 = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, α1)
    state2 = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, α2)
    coeffs = Float32[0.6, 0.8]
    lc = GaussianLinearCombination(basis, coeffs, [state1, state2])
    @test device(lc) == :gpu
    @test length(lc) == 2
    
    @testset "Batched Wigner Evaluation" begin
        num_points = 100
        x_points = CuArray(randn(Float32, 2, num_points))
        w_batch = wigner(lc, x_points)
        @test w_batch isa CuArray{Float32}
        @test size(w_batch) == (num_points,)
        @test device(w_batch) == :gpu
        @test all(isfinite, Array(w_batch))
        w_single = [wigner(lc, @view x_points[:, i]) for i in 1:min(10, num_points)]
        w_batch_sample = Array(w_batch[1:length(w_single)])
        @test w_batch_sample ≈ w_single rtol=1e-5
    end
    
    @testset "Batched Wigner Characteristic Function" begin  
        num_points = 50
        xi_points = CuArray(randn(Float32, 2, num_points))
        char_batch = wignerchar(lc, xi_points)
        @test char_batch isa CuArray{ComplexF32}
        @test size(char_batch) == (num_points,)
        @test device(char_batch) == :gpu
        @test all(isfinite, Array(real.(char_batch)))
        @test all(isfinite, Array(imag.(char_batch)))
        char_single = [wignerchar(lc, @view xi_points[:, i]) for i in 1:min(5, num_points)]
        char_batch_sample = Array(char_batch[1:length(char_single)])
        @test char_batch_sample ≈ char_single rtol=1e-5
    end
    
    @testset "Automatic CPU->GPU Promotion" begin
        num_points = 20
        x_points_cpu = randn(Float32, 2, num_points)
        xi_points_cpu = randn(Float32, 2, num_points)
        w_auto = wigner(lc, x_points_cpu)
        char_auto = wignerchar(lc, xi_points_cpu)
        @test device(w_auto) == :gpu
        @test device(char_auto) == :gpu
        w_direct = wigner(lc, CuArray(x_points_cpu))
        char_direct = wignerchar(lc, CuArray(xi_points_cpu))
        @test Array(w_auto) ≈ Array(w_direct) rtol=1e-6
        @test Array(char_auto) ≈ Array(char_direct) rtol=1e-6
    end
    
    @testset "Performance Scaling" begin
        large_num_points = 1000
        x_large = CuArray(randn(Float32, 2, large_num_points))
        @test_nowarn begin
            w_large = wigner(lc, x_large)
            @test length(w_large) == large_num_points
            @test device(w_large) == :gpu
        end
    end
    
    @testset "Multi-State Interference" begin
        state3 = squeezedstate(CuVector{Float32}, CuMatrix{Float32}, basis, 0.2f0, Float32(π/6))
        coeffs_multi = Float32[0.5, 0.3, 0.2]
        lc_multi = GaussianLinearCombination(basis, coeffs_multi, [state1, state2, state3])
        num_points = 50
        x_points = CuArray(randn(Float32, 2, num_points))
        w_multi = wigner(lc_multi, x_points)
        char_multi = wignerchar(lc_multi, x_points)
        @test w_multi isa CuArray{Float32}
        @test char_multi isa CuArray{ComplexF32}
        @test size(w_multi) == (num_points,)
        @test size(char_multi) == (num_points,)
        @test all(isfinite, Array(w_multi))
        @test all(isfinite, Array(real.(char_multi)))
        @test all(isfinite, Array(imag.(char_multi)))
    end
    
    @testset "Error Handling and Edge Cases" begin
        x_wrong_dim = CuArray(randn(Float32, 3, 10))
        @test_throws ArgumentError wigner(lc, x_wrong_dim)
        @test_throws ArgumentError wignerchar(lc, x_wrong_dim)
        @test_throws DimensionMismatch GaussianLinearCombination(basis, Float32[0.5, 0.3], [state1])
        basis_wrong = QuadBlockBasis(1)
        state_wrong = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis_wrong, 1.0f0)
        @test_throws ArgumentError GaussianLinearCombination(basis, Float32[0.5], [state_wrong])
        x_single = CuArray(randn(Float32, 2, 1))
        w_single_batch = wigner(lc, x_single)
        @test length(w_single_batch) == 1
        @test isfinite(Array(w_single_batch)[1])
    end
end

@testset "GPU Batched Interference - Float64 Precision" begin
    basis = QuadPairBasis(1)
    state1_f64 = coherentstate(CuVector{Float64}, CuMatrix{Float64}, basis, 1.0 + 0.5*im)
    state2_f64 = coherentstate(CuVector{Float64}, CuMatrix{Float64}, basis, -1.0 + 0.3*im)
    coeffs_f64 = [0.6, 0.8]
    lc_f64 = GaussianLinearCombination(basis, coeffs_f64, [state1_f64, state2_f64])
    x_points_f64 = CuArray(randn(Float64, 2, 20))
    w_f64 = wigner(lc_f64, x_points_f64)
    char_f64 = wignerchar(lc_f64, x_points_f64)
    @test w_f64 isa CuArray{Float64}
    @test char_f64 isa CuArray{ComplexF64}
    @test all(isfinite, Array(w_f64))
    @test all(isfinite, Array(real.(char_f64)))
    @test all(isfinite, Array(imag.(char_f64)))
end

@testset "GPU Batched Interference - Multi-Mode" begin
    basis_2mode = QuadPairBasis(2)
    state1_2m = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis_2mode, 1.0f0)
    state2_2m = squeezedstate(CuVector{Float32}, CuMatrix{Float32}, basis_2mode, 0.3f0, 0.0f0)
    coeffs_2m = Float32[0.7, 0.3]
    lc_2m = GaussianLinearCombination(basis_2mode, coeffs_2m, [state1_2m, state2_2m])
    num_points = 30
    x_points_2m = CuArray(randn(Float32, 4, num_points))
    w_2m = wigner(lc_2m, x_points_2m)
    char_2m = wignerchar(lc_2m, x_points_2m)
    @test w_2m isa CuArray{Float32}
    @test char_2m isa CuArray{ComplexF32}
    @test size(w_2m) == (num_points,)
    @test size(char_2m) == (num_points,)
    @test all(isfinite, Array(w_2m))
    @test all(isfinite, Array(real.(char_2m)))
    @test all(isfinite, Array(imag.(char_2m)))
end

end

#Benchmark code
#=
@testitem "GPU Performance Benchmarks for Wigner and Tensor" begin
    using CUDA
    using Gabs
    using Gabs: device
    using LinearAlgebra
    using Statistics
    
    function gpu_warmup()
        basis = QuadPairBasis(1)
        try
            for _ in 1:5
                vac = vacuumstate(basis) |> gpu
                coh = coherentstate(basis, 1.0f0) |> gpu
            end
            vac = vacuumstate(basis) |> gpu
            disp = displace(basis, 1.0f0) |> gpu
            for _ in 1:5
                displaced = disp * vac
            end
            coh = coherentstate(basis, 0.5f0) |> gpu
            test_points = CuArray(randn(Float32, 2, 100))
            for _ in 1:3
                w = wigner(coh, test_points)
            end
            CUDA.synchronize()
            GC.gc()
        catch e
            @error "GPU warmup failed" exception=e
            rethrow(e)
        end
    end
    
    function benchmark_operation(gpu_func, cpu_func, name::String; n_trials=5, min_time=1e-6)
        gpu_times = Float64[]
        for trial in 1:n_trials
            CUDA.synchronize()
            t_start = time_ns()
            result_gpu = gpu_func()
            CUDA.synchronize()
            t_end = time_ns()
            elapsed = max((t_end - t_start) / 1e9, min_time)
            push!(gpu_times, elapsed)
        end
        cpu_times = Float64[]
        for trial in 1:n_trials
            GC.gc()
            t_start = time_ns()
            result_cpu = cpu_func()
            t_end = time_ns()
            elapsed = max((t_end - t_start) / 1e9, min_time)
            push!(cpu_times, elapsed)
        end
        gpu_median = median(gpu_times)
        cpu_median = median(cpu_times)
        speedup = cpu_median / gpu_median
        @info "Benchmark: $name" cpu_time=round(cpu_median, digits=4) gpu_time=round(gpu_median, digits=4) speedup=round(speedup, digits=2)
        return speedup, gpu_median, cpu_median
    end
    
    @testset "GPU Warmup" begin
        gpu_warmup()
        @test true
    end
    
    @testset "API Usage" begin
        basis = QuadPairBasis(1)
        state = coherentstate(basis, 1.0) |> gpu
        op = displace(basis, 0.5) |> gpu
        result = op * state
        @test device(state) == :gpu
        @test device(op) == :gpu
        @test device(result) == :gpu
        α_gpu = CuArray([1.0f0 + 0.5f0im])
        auto_state = coherentstate(basis, α_gpu)
        @test device(auto_state) == :gpu
    end
    
    @testset "State Creation Benchmarks" begin
        
        @testset "Single-Mode State Creation" begin
            basis = QuadPairBasis(1)
            α = 1.0f0 + 0.5f0im
            gpu_func = () -> coherentstate(basis, α) |> gpu
            cpu_func = () -> coherentstate(basis, α)
            speedup, _, _ = benchmark_operation(gpu_func, cpu_func, "Single-mode coherent state creation")
            coh_gpu = gpu_func()
            coh_cpu = cpu_func()
            @test Array(coh_gpu.mean) ≈ Float32.(coh_cpu.mean) rtol=1e-6
            @test speedup > 0.01
        end
        
        @testset "Multi-Mode State Creation" begin
            basis = QuadPairBasis(10)
            α_vec = randn(ComplexF32, 10)
            gpu_func = () -> coherentstate(basis, α_vec) |> gpu
            cpu_func = () -> coherentstate(basis, α_vec)
            speedup, _, _ = benchmark_operation(gpu_func, cpu_func, "10-mode coherent state creation")
            @test speedup > 0.03
        end
        
        @testset "Very Large State Creation" begin
            basis = QuadPairBasis(20)
            α_vec = randn(ComplexF32, 20)
            gpu_func = () -> coherentstate(basis, α_vec) |> gpu
            cpu_func = () -> coherentstate(basis, α_vec)
            speedup, _, _ = benchmark_operation(gpu_func, cpu_func, "20-mode coherent state creation")
            @test speedup > 0.04
            if speedup > 1.0
                @info "GPU advantage emerging for very large state creation: $(round(speedup, digits=2))x"
            end
        end
    end
    
    @testset "Operation Application Benchmarks" begin
        
        @testset "Single-Mode Operations" begin
            basis = QuadPairBasis(1)
            vac_gpu = vacuumstate(basis) |> gpu
            vac_cpu = vacuumstate(basis)
            disp_gpu = displace(basis, 1.0f0) |> gpu
            disp_cpu = displace(basis, 1.0f0)
            gpu_func = () -> disp_gpu * vac_gpu
            cpu_func = () -> disp_cpu * vac_cpu
            speedup, _, _ = benchmark_operation(gpu_func, cpu_func, "Single-mode operation application")
            result_gpu = gpu_func()
            result_cpu = cpu_func()
            @test Array(result_gpu.mean) ≈ Float32.(result_cpu.mean) rtol=1e-6
            @test speedup > 0.01
        end
        
        @testset "Multi-Mode Operations" begin
            basis = QuadPairBasis(5)
            α_vec = randn(ComplexF32, 5)
            coh_gpu = coherentstate(basis, α_vec) |> gpu
            coh_cpu = coherentstate(basis, α_vec)
            r_vec = randn(Float32, 5) * 0.3f0
            θ_vec = randn(Float32, 5)
            squeeze_gpu = squeeze(basis, r_vec, θ_vec) |> gpu
            squeeze_cpu = squeeze(basis, r_vec, θ_vec)
            gpu_func = () -> squeeze_gpu * coh_gpu
            cpu_func = () -> squeeze_cpu * coh_cpu
            speedup, _, _ = benchmark_operation(gpu_func, cpu_func, "5-mode squeeze operation")
            @test speedup > 0.02
        end
    end
    
    @testset "Tensor Product Benchmarks" begin
        
        @testset "Small Tensor Products" begin
            basis1 = QuadPairBasis(2)
            basis2 = QuadPairBasis(2)
            α1 = randn(ComplexF32, 2)
            α2 = randn(ComplexF32, 2)
            coh1_gpu = coherentstate(basis1, α1) |> gpu
            coh2_gpu = coherentstate(basis2, α2) |> gpu
            coh1_cpu = coherentstate(basis1, α1)
            coh2_cpu = coherentstate(basis2, α2)
            gpu_func = () -> tensor(coh1_gpu, coh2_gpu)
            cpu_func = () -> tensor(coh1_cpu, coh2_cpu)
            speedup, _, _ = benchmark_operation(gpu_func, cpu_func, "Small tensor product (2⊗2 modes)")
            result_gpu = gpu_func()
            result_cpu = cpu_func()
            @test result_gpu.basis.nmodes == 4
            @test Array(result_gpu.mean) ≈ Float32.(result_cpu.mean) rtol=1e-6
            @test Array(result_gpu.covar) ≈ Float32.(result_cpu.covar) rtol=1e-6
            @test speedup > 0.05
        end
        
        @testset "Larger Tensor Products" begin
            basis1 = QuadPairBasis(3)
            basis2 = QuadPairBasis(4)
            α1 = randn(ComplexF32, 3)
            α2 = randn(ComplexF32, 4)
            coh1_gpu = coherentstate(basis1, α1) |> gpu
            coh2_gpu = coherentstate(basis2, α2) |> gpu
            coh1_cpu = coherentstate(basis1, α1)
            coh2_cpu = coherentstate(basis2, α2)
            gpu_func = () -> tensor(coh1_gpu, coh2_gpu)
            cpu_func = () -> tensor(coh1_cpu, coh2_cpu)
            speedup, _, _ = benchmark_operation(gpu_func, cpu_func, "Large tensor product (3⊗4 modes)")
            result_gpu = gpu_func()
            result_cpu = cpu_func()
            @test result_gpu.basis.nmodes == 7
            @test Array(result_gpu.mean) ≈ Float32.(result_cpu.mean) rtol=1e-6
            @test speedup > 0.05
            if speedup > 1.0
                @info "GPU tensor product advantage: $(round(speedup, digits=2))x"
            end
        end
    end
    
    @testset "Wigner Function Benchmarks - The GPU Sweet Spot" begin
        
        @testset "Single-Point Wigner" begin
            basis = QuadPairBasis(1)
            coh_gpu = coherentstate(basis, 1.0f0) |> gpu
            coh_cpu = coherentstate(basis, 1.0f0)
            x_point = [0.5f0, 0.3f0]
            gpu_func = () -> wigner(coh_gpu, x_point)
            cpu_func = () -> wigner(coh_cpu, x_point)
            speedup, _, _ = benchmark_operation(gpu_func, cpu_func, "Single-point Wigner evaluation")
            @test gpu_func() ≈ cpu_func() rtol=1e-5
            @test speedup > 0.05
        end
        
        @testset "Batch Wigner Evaluation - Small" begin
            basis = QuadPairBasis(1)
            coh_gpu = coherentstate(basis, 0.8f0) |> gpu
            coh_cpu = coherentstate(basis, 0.8f0)
            n_points = 1000
            x_points_gpu = CuArray(randn(Float32, 2, n_points))
            x_points_cpu = Array(x_points_gpu)
            gpu_func = () -> wigner(coh_gpu, x_points_gpu)
            cpu_func = () -> [wigner(coh_cpu, x_points_cpu[:, i]) for i in 1:n_points]
            speedup, _, _ = benchmark_operation(gpu_func, cpu_func, "Batch Wigner (1000 points)", n_trials=3)
            @test speedup > 2.0
            w_gpu = Array(gpu_func())
            w_cpu = cpu_func()
            @test w_gpu ≈ w_cpu rtol=1e-4
            @info "GPU batch Wigner showing advantage: $(round(speedup, digits=1))x speedup"
        end
        
        @testset "Batch Wigner Evaluation - Large" begin
            basis = QuadPairBasis(1)
            coh_gpu = coherentstate(basis, 0.5f0) |> gpu
            coh_cpu = coherentstate(basis, 0.5f0)
            n_points = 25000
            x_points_gpu = CuArray(randn(Float32, 2, n_points))
            x_points_cpu = Array(x_points_gpu)
            gpu_func = () -> wigner(coh_gpu, x_points_gpu)
            cpu_func = () -> [wigner(coh_cpu, x_points_cpu[:, i]) for i in 1:n_points]
            speedup, gpu_time, cpu_time = benchmark_operation(gpu_func, cpu_func, "Large batch Wigner (25,000 points)", n_trials=3)
            @test speedup > 10.0
            @test gpu_time < 0.1
            w_gpu = Array(gpu_func())
            for i in 1:min(10, n_points)
                w_single = wigner(coh_cpu, x_points_cpu[:, i])
                @test w_gpu[i] ≈ w_single rtol=1e-4
            end
            @info "Large batch GPU advantage: $(round(speedup, digits=1))x speedup"
        end
        
        @testset "Multi-Mode Wigner" begin
            basis = QuadPairBasis(2)
            α_vec = [0.5f0 + 0.3f0im, -0.2f0 + 0.8f0im]
            coh_gpu = coherentstate(basis, α_vec) |> gpu
            coh_cpu = coherentstate(basis, α_vec)
            n_points = 5000
            x_points_gpu = CuArray(randn(Float32, 4, n_points))
            x_points_cpu = Array(x_points_gpu)
            gpu_func = () -> wigner(coh_gpu, x_points_gpu)
            cpu_func = () -> [wigner(coh_cpu, x_points_cpu[:, i]) for i in 1:n_points]
            speedup, _, _ = benchmark_operation(gpu_func, cpu_func, "Multi-mode Wigner (2 modes, 5000 points)", n_trials=3)
            @test speedup > 5.0
            @info "Multi-mode GPU advantage: $(round(speedup, digits=1))x speedup"
        end
    end
    
    @testset "Wigner Characteristic Function Benchmarks" begin
        
        @testset "Batch Wigner Characteristic - Professional API" begin
            basis = QuadPairBasis(1)
            sq_gpu = squeezedstate(basis, 0.3f0, Float32(π/4)) |> gpu
            sq_cpu = squeezedstate(basis, 0.3f0, Float32(π/4))
            n_points = 5000
            xi_points_gpu = CuArray(randn(Float32, 2, n_points) * 0.5f0)
            xi_points_cpu = Array(xi_points_gpu)
            gpu_func = () -> wignerchar(sq_gpu, xi_points_gpu)
            cpu_func = () -> [wignerchar(sq_cpu, xi_points_cpu[:, i]) for i in 1:n_points]
            speedup, _, _ = benchmark_operation(gpu_func, cpu_func, "Batch Wigner characteristic (5,000 points)", n_trials=3)
            @test speedup > 2.0
            chi_gpu = Array(gpu_func())
            for i in 1:min(5, n_points)
                chi_single = wignerchar(sq_cpu, xi_points_cpu[:, i])
                @test real(chi_gpu[i]) ≈ real(chi_single) rtol=1e-4
                @test imag(chi_gpu[i]) ≈ imag(chi_single) rtol=1e-4
            end
            @info "Wigner characteristic GPU advantage: $(round(speedup, digits=1))x speedup"
        end
    end
    
    @testset " Performance Tests" begin
        basis = QuadPairBasis(1)
        coh_gpu = coherentstate(basis, 1.0f0) |> gpu
        coh_cpu = coherentstate(basis, 1.0f0)
        n_points = 15000
        x_points_gpu = CuArray(randn(Float32, 2, n_points))
        x_points_cpu = Array(x_points_gpu)
        gpu_func = () -> wigner(coh_gpu, x_points_gpu)
        cpu_func = () -> [wigner(coh_cpu, x_points_cpu[:, i]) for i in 1:n_points]
        speedup, gpu_time, cpu_time = benchmark_operation(gpu_func, cpu_func, " GPU validation")
        @test speedup > 15.0
        @test gpu_time < cpu_time / 10  
        w_gpu = Array(gpu_func())
        w_sample = [wigner(coh_cpu, x_points_cpu[:, i]) for i in 1:min(20, n_points)]
        @test w_gpu[1:length(w_sample)] ≈ w_sample rtol=1e-4
    end
    
    @testset "Scaling Analysis" begin
        basis = QuadPairBasis(1)
        coh_gpu = coherentstate(basis, 1.0f0) |> gpu
        coh_cpu = coherentstate(basis, 1.0f0)
        point_counts = [100, 500, 1000, 5000, 10000, 25000]
        speedups = Float64[]
        for n_points in point_counts
            x_points_gpu = CuArray(randn(Float32, 2, n_points))
            x_points_cpu = Array(x_points_gpu)
            gpu_func = () -> wigner(coh_gpu, x_points_gpu)
            cpu_func = () -> [wigner(coh_cpu, x_points_cpu[:, i]) for i in 1:n_points]
            speedup, gpu_time, cpu_time = benchmark_operation(gpu_func, cpu_func, "Scaling test ($n_points points)", n_trials=3)
            push!(speedups, speedup)
            @info "Scaling point" n_points=n_points speedup=round(speedup, digits=2) gpu_time=round(gpu_time, digits=4) cpu_time=round(cpu_time, digits=4)
        end
        @test speedups[end] > speedups[1]  
        @test maximum(speedups) > 10.0    
        @info "Peak GPU speedup: $(round(maximum(speedups), digits=2))x"
    end
end
=#

#=
@testitem "GPU Batched Performance Benchmarks" begin
using Gabs
using Gabs: device
using LinearAlgebra  
using CUDA
using Test
using BenchmarkTools

function cpu_wigner_batch(lc, x_points)
    results = Vector{Float32}(undef, size(x_points, 2))
    for i in 1:size(x_points, 2)
        results[i] = wigner(lc, x_points[:, i])
    end
    return results
end

function cpu_wignerchar_batch(lc, xi_points)
    results = Vector{ComplexF32}(undef, size(xi_points, 2))
    for i in 1:size(xi_points, 2)
        results[i] = wignerchar(lc, xi_points[:, i])
    end
    return results
end

function cpu_lc_wigner_batch(lc, x_points)
    results = Vector{Float32}(undef, size(x_points, 2))
    for i in 1:size(x_points, 2)
        results[i] = wigner(lc, x_points[:, i])
    end
    return results
end

@testset "GPU vs CPU Speedup Benchmarks - Batched Interference" begin
    println("GPU BATCHED INTERFERENCE PERFORMANCE BENCHMARKS")
    test_configs = [
        (1, 100,    "Small: 1-mode, 2-state, 100 points"),
        (1, 1_000,  "Medium: 1-mode, 2-state, 1K points"), 
        (1, 10_000, "Large: 1-mode, 2-state, 10K points"),
        (2, 1_000,  "Multi-mode: 2-mode, 2-state, 1K points"),
        (1, 50_000, "Very Large: 1-mode, 2-state, 50K points"),
        (3, 500,    "High-D: 3-mode, 2-state, 500 points"),
    ]
    @testset "Wigner Function Benchmarks" begin
        println("\nWIGNER FUNCTION BENCHMARKS:")        
        for (nmodes, npoints, description) in test_configs
            println("\nTesting: $description")
            basis = QuadPairBasis(nmodes)
            phase_space_dim = 2 * nmodes
            state1_cpu = coherentstate(basis, 1.0f0 + 0.5f0*im)
            state2_cpu = coherentstate(basis, -1.0f0 + 0.3f0*im)
            state1_gpu = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, 1.0f0 + 0.5f0*im)
            state2_gpu = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, -1.0f0 + 0.3f0*im)
            coeffs = Float32[0.6, 0.8]
            coeffs ./= norm(coeffs)
            cpu_states_array = [state1_cpu, state2_cpu]
            gpu_states_array = [state1_gpu, state2_gpu]
            cpu_lc = GaussianLinearCombination(basis, coeffs, cpu_states_array)
            gpu_lc = GaussianLinearCombination(basis, coeffs, gpu_states_array)
            @test device(gpu_lc) == :gpu
            x_points_cpu = randn(Float32, phase_space_dim, npoints)
            x_points_gpu = CuArray(x_points_cpu)
            if npoints >= 100
                wigner(cpu_lc, x_points_cpu[:, 1])
                wigner(gpu_lc, x_points_gpu[:, 1:min(10, npoints)])
                CUDA.synchronize()
            end
            print("   CPU: ")
            cpu_time = @belapsed cpu_wigner_batch($cpu_lc, $x_points_cpu) samples=3 evals=1
            cpu_result = cpu_wigner_batch(cpu_lc, x_points_cpu)
            print("   GPU: ")
            gpu_time = @belapsed begin
                result = wigner($gpu_lc, $x_points_gpu)
                CUDA.synchronize()
                result
            end samples=3 evals=1
            gpu_result = wigner(gpu_lc, x_points_gpu)
            CUDA.synchronize()
            sample_size = min(20, npoints)
            cpu_sample = cpu_result[1:sample_size]
            gpu_sample = Array(gpu_result[1:sample_size])
            @test cpu_sample ≈ gpu_sample rtol=1e-4
            speedup = cpu_time / gpu_time
            println("   Results:")
            println("      CPU Time: $(round(cpu_time*1000, digits=2)) ms")
            println("      GPU Time: $(round(gpu_time*1000, digits=2)) ms")
            println("      Speedup:  $(round(speedup, digits=1))x")
        end
    end
    
    @testset "Wigner Characteristic Function Benchmarks" begin
        println("\n\nWIGNER CHARACTERISTIC FUNCTION BENCHMARKS:")
        char_configs = [
            (1, 500,   "Small: 1-mode, 2-state, 500 points"),
            (1, 5_000, "Medium: 1-mode, 2-state, 5K points"),
            (2, 1_000, "Multi-mode: 2-mode, 2-state, 1K points"),
            (1, 20_000,"Large: 1-mode, 2-state, 20K points"),
        ]
        
        for (nmodes, npoints, description) in char_configs
            println("\nTesting: $description")
            basis = QuadPairBasis(nmodes)
            phase_space_dim = 2 * nmodes
            state1_cpu = coherentstate(basis, 1.0f0 + 0.5f0*im)
            state2_cpu = coherentstate(basis, -1.0f0 + 0.3f0*im)
            state1_gpu = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, 1.0f0 + 0.5f0*im)
            state2_gpu = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, -1.0f0 + 0.3f0*im)
            coeffs = Float32[0.6, 0.8]
            coeffs ./= norm(coeffs)
            cpu_states_array = [state1_cpu, state2_cpu]
            gpu_states_array = [state1_gpu, state2_gpu]
            cpu_lc = GaussianLinearCombination(basis, coeffs, cpu_states_array)
            gpu_lc = GaussianLinearCombination(basis, coeffs, gpu_states_array)
            xi_points_cpu = randn(Float32, phase_space_dim, npoints)
            xi_points_gpu = CuArray(xi_points_cpu)
            if npoints >= 100
                wignerchar(cpu_lc, xi_points_cpu[:, 1])
                wignerchar(gpu_lc, xi_points_gpu[:, 1:min(5, npoints)])
                CUDA.synchronize()
            end
            print("   CPU: ")
            cpu_time = @belapsed cpu_wignerchar_batch($cpu_lc, $xi_points_cpu) samples=3 evals=1
            cpu_result = cpu_wignerchar_batch(cpu_lc, xi_points_cpu)
            print("   GPU: ")
            gpu_time = @belapsed begin
                result = wignerchar($gpu_lc, $xi_points_gpu)
                CUDA.synchronize()
                result
            end samples=3 evals=1
            gpu_result = wignerchar(gpu_lc, xi_points_gpu)
            CUDA.synchronize()
            sample_size = min(10, npoints)
            cpu_sample = cpu_result[1:sample_size]
            gpu_sample = Array(gpu_result[1:sample_size])
            @test cpu_sample ≈ gpu_sample rtol=1e-4
            speedup = cpu_time / gpu_time
            println("   Results:")
            println("      CPU Time: $(round(cpu_time*1000, digits=2)) ms")
            println("      GPU Time: $(round(gpu_time*1000, digits=2)) ms")
            println("      Speedup:  $(round(speedup, digits=1))x")
        end
    end
    
    @testset "Cross-Wigner Batch Performance" begin
        println("\n\nCROSS-WIGNER BATCH PERFORMANCE:")
        basis = QuadPairBasis(1)
        state1_cpu = coherentstate(basis, 1.0f0 + 0.5f0*im)
        state2_cpu = squeezedstate(basis, 0.3f0, Float32(π/4))
        state1_gpu = coherentstate(CuVector{Float32}, CuMatrix{Float32}, basis, 1.0f0 + 0.5f0*im)
        state2_gpu = squeezedstate(CuVector{Float32}, CuMatrix{Float32}, basis, 0.3f0, Float32(π/4))
        batch_sizes = [100, 1_000, 10_000, 50_000]
        
        for npoints in batch_sizes
            println("\nCross-Wigner batch size: $npoints points")
            x_points_cpu = randn(Float32, 2, npoints)
            x_points_gpu = CuArray(x_points_cpu)
            if npoints >= 100
                Gabs.cross_wigner(state1_cpu, state2_cpu, x_points_cpu[:, 1])
            end
            coeffs = Float32[0.7, 0.3]
            cpu_states_array = [state1_cpu, state2_cpu]
            gpu_states_array = [state1_gpu, state2_gpu]
            cpu_lc = GaussianLinearCombination(basis, coeffs, cpu_states_array)
            gpu_lc = GaussianLinearCombination(basis, coeffs, gpu_states_array)
            print("   CPU: ")
            cpu_time = @belapsed cpu_lc_wigner_batch($cpu_lc, $x_points_cpu) samples=2 evals=1
            print("   GPU: ")
            gpu_time = @belapsed begin
                result = wigner($gpu_lc, $x_points_gpu)
                CUDA.synchronize()
                result
            end samples=2 evals=1
            speedup = cpu_time / gpu_time
            println("   Results:")
            println("      CPU Time: $(round(cpu_time*1000, digits=2)) ms")
            println("      GPU Time: $(round(gpu_time*1000, digits=2)) ms") 
            println("      Speedup:  $(round(speedup, digits=1))x")
        end
    end
end
end
=#