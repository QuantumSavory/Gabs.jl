"""
    cross_wigner_kernel!(result, x_point, mean1, mean2, avg_covar_inv, 
                         omega, log_det_avg_covar, nmodes, hbar)

GPU kernel for computing cross-Wigner function between two Gaussian states.
Implements: W₁₂(x) = norm × exp[-½(x-μ̄)ᵀV̄⁻¹(x-μ̄)] × exp[i(μ₁-μ₂)ᵀΩ(x-μ̄)/ħ]
where μ̄ = (μ₁+μ₂)/2, V̄ = (V₁+V₂)/2
"""
function cross_wigner_kernel!(result::CuDeviceArray{Complex{T}}, 
                              x_point::CuDeviceVector{T},
                              mean1::CuDeviceVector{T}, 
                              mean2::CuDeviceVector{T},
                              avg_covar_inv::CuDeviceMatrix{T},
                              omega::CuDeviceMatrix{T}, 
                              log_det_avg_covar::T, 
                              nmodes::Int, hbar::T) where T

    if threadIdx().x == 1 && blockIdx().x == 1
        quadratic_form = zero(T)
        for i in 1:length(mean1)
            avg_mean_i = (mean1[i] + mean2[i]) * T(0.5)
            dx_i = x_point[i] - avg_mean_i
            for j in 1:length(mean1)
                avg_mean_j = (mean1[j] + mean2[j]) * T(0.5)
                dx_j = x_point[j] - avg_mean_j
                quadratic_form += dx_i * avg_covar_inv[i, j] * dx_j
            end
        end
        phase_arg = zero(T)
        for i in 1:length(mean1)
            mu_diff_i = mean1[i] - mean2[i]
            avg_mean_i = (mean1[i] + mean2[i]) * T(0.5)
            dx_i = x_point[i] - avg_mean_i
            omega_dx_i = zero(T)
            for j in 1:length(mean1)
                avg_mean_j = (mean1[j] + mean2[j]) * T(0.5)
                dx_j = x_point[j] - avg_mean_j
                omega_dx_i += omega[i, j] * dx_j
            end
            phase_arg += mu_diff_i * omega_dx_i
        end
        log_normalization = -T(nmodes) * log(T(2π)) - T(0.5) * log_det_avg_covar
        normalization = exp(log_normalization)
        gaussian_part = exp(-T(0.5) * quadratic_form)
        phase_factor = Complex{T}(cos(phase_arg / hbar), sin(phase_arg / hbar))
        result[1] = normalization * gaussian_part * phase_factor
    end
    return nothing
end

"""
    cross_wignerchar_kernel!(result, xi_point, quadratic_matrix, linear_vector)

GPU kernel for computing cross-Wigner characteristic function between two Gaussian states.
Implements: χ₁₂(ξ) = exp(-½ξᵀΩV̄Ωᵀξ - iΩμ̄·ξ)
where μ̄ = (μ₁+μ₂)/2, V̄ = (V₁+V₂)/2
"""
function cross_wignerchar_kernel!(result::CuDeviceArray{Complex{T}},
                                  xi_point::CuDeviceVector{T},
                                  quadratic_matrix::CuDeviceMatrix{T},
                                  linear_vector::CuDeviceVector{T}) where T
    
    if threadIdx().x == 1 && blockIdx().x == 1
        quadratic_term = zero(T)
        for i in 1:length(xi_point)
            for j in 1:length(xi_point)
                quadratic_term += xi_point[i] * quadratic_matrix[i, j] * xi_point[j]
            end
        end
        linear_term = zero(T)
        for i in 1:length(xi_point)
            linear_term += linear_vector[i] * xi_point[i]
        end
        real_part = -T(0.5) * quadratic_term
        imag_part = -linear_term
        result[1] = Complex{T}(cos(imag_part), sin(imag_part)) * exp(real_part)
    end
    return nothing
end

"""
    gpu_safe_logdet(A::CuMatrix{T}) where T

Compute log determinant of positive definite matrix using Cholesky decomposition.
"""
function gpu_safe_logdet(A::CuMatrix{T}) where T
    try
        L = cholesky(Symmetric(A)).L
        diag_L = diag(L)  
        log_det_L = sum(log.(diag_L))  
        return T(2) * log_det_L
    catch e
        A_cpu = Array(A)
        return T(logdet(A_cpu))
    end
end

"""
    gpu_safe_inv(A::CuMatrix{T}) where T

Compute matrix inverse using GPU-safe operations.
"""
function gpu_safe_inv(A::CuMatrix{T}) where T
    try
        return inv(A)  
    catch e
        A_cpu = Array(A)
        return CuArray{T}(inv(A_cpu))
    end
end

"""
    gpu_safe_length_check(x::AbstractVector, expected_len::Int)
"""
function gpu_safe_length_check(x::AbstractVector, expected_len::Int)
    actual_len = size(x, 1)  
    return actual_len == expected_len
end

function Gabs.cross_wigner(state1::GaussianState{<:SymplecticBasis,<:CuArray,<:CuArray}, 
                           state2::GaussianState{<:SymplecticBasis,<:CuArray,<:CuArray}, 
                           x::AbstractVector)
    
    if !CUDA_AVAILABLE
        @warn "CUDA not available for cross_wigner. This should not happen in CUDAExt."
        error("GPU cross_wigner called but CUDA not available")
    end
    typeof(state1.basis) == typeof(state2.basis) || throw(ArgumentError(SYMPLECTIC_ERROR))
    state1.basis == state2.basis || throw(ArgumentError(SYMPLECTIC_ERROR))
    state1.ħ == state2.ħ || throw(ArgumentError(HBAR_ERROR))
    expected_len = size(state1.mean, 1)
    actual_len = size(x, 1)
    actual_len == expected_len || throw(ArgumentError(WIGNER_ERROR))
    if state1 === state2
        wigner_result = wigner(state1, x)
        T = real(eltype(state1.mean))
        return Complex{T}(wigner_result, zero(T))
    end
    T = real(eltype(state1.mean))
    nmodes = state1.basis.nmodes
    x_gpu = CuArray{T}(x)
    avg_covar = (state1.covar .+ state2.covar) ./ T(2)
    avg_covar_inv = gpu_safe_inv(avg_covar)
    log_det_avg_covar = gpu_safe_logdet(avg_covar)
    omega_cpu = symplecticform(state1.basis)
    omega_gpu = CuArray{T}(omega_cpu)
    result = CuArray{Complex{T}}(undef, 1)
    @cuda threads=1 blocks=1 cross_wigner_kernel!(
        result, x_gpu, state1.mean, state2.mean, avg_covar_inv, 
        omega_gpu, log_det_avg_covar, nmodes, T(state1.ħ)
    )
    CUDA.synchronize()  
    return Array(result)[1]  
end

function Gabs.cross_wignerchar(state1::GaussianState{<:SymplecticBasis,<:CuArray,<:CuArray}, 
                               state2::GaussianState{<:SymplecticBasis,<:CuArray,<:CuArray}, 
                               xi::AbstractVector)
    
    if !CUDA_AVAILABLE
        @warn "CUDA not available for cross_wignerchar. This should not happen in CUDAExt."
        error("GPU cross_wignerchar called but CUDA not available")
    end
    typeof(state1.basis) == typeof(state2.basis) || throw(ArgumentError(SYMPLECTIC_ERROR))
    state1.basis == state2.basis || throw(ArgumentError(SYMPLECTIC_ERROR))
    state1.ħ == state2.ħ || throw(ArgumentError(HBAR_ERROR))
    expected_len = size(state1.mean, 1)
    actual_len = size(xi, 1)
    actual_len == expected_len || throw(ArgumentError(WIGNER_ERROR))
    if state1 === state2
        return wignerchar(state1, xi)
    end
    T = real(eltype(state1.mean))
    xi_gpu = CuArray{T}(xi)
    avg_mean = (state1.mean .+ state2.mean) ./ T(2)
    avg_covar = (state1.covar .+ state2.covar) ./ T(2)
    omega_cpu = symplecticform(state1.basis)
    omega_gpu = CuArray{T}(omega_cpu)
    temp_matrix = omega_gpu * avg_covar
    quadratic_matrix = temp_matrix * transpose(omega_gpu)
    linear_vector = omega_gpu * avg_mean
    result = CuArray{Complex{T}}(undef, 1)
    @cuda threads=1 blocks=1 cross_wignerchar_kernel!(
        result, xi_gpu, quadratic_matrix, linear_vector
    )
    CUDA.synchronize()  
    return Array(result)[1]  
end

function Gabs.cross_wigner(state1::GaussianState{<:SymplecticBasis,<:CuArray,<:CuArray}, 
                           state2::GaussianState{<:SymplecticBasis,<:CuArray,<:CuArray}, 
                           x::Vector{T}) where T<:Real
    return cross_wigner(state1, state2, CuArray{real(eltype(state1.mean))}(x))
end

function Gabs.cross_wignerchar(state1::GaussianState{<:SymplecticBasis,<:CuArray,<:CuArray}, 
                               state2::GaussianState{<:SymplecticBasis,<:CuArray,<:CuArray}, 
                               xi::Vector{T}) where T<:Real
    return cross_wignerchar(state1, state2, CuArray{real(eltype(state1.mean))}(xi))
end