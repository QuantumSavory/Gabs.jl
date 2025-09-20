"""
    wigner_kernel!(output, x_points, mean, covar_inv, det_covar, nmodes)

CUDA kernel for computing Wigner function values at multiple phase space points.
"""
function wigner_kernel!(output::CuDeviceVector{T}, x_points::CuDeviceMatrix{T}, 
                       mean::CuDeviceVector{T}, covar_inv::CuDeviceMatrix{T}, 
                       det_covar::T, nmodes::Int) where T
    
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx <= size(x_points, 2)
        diff_sum = zero(T)
        
        for i in 1:length(mean)
            diff_i = x_points[i, idx] - mean[i]
            temp_sum = zero(T)
            
            for j in 1:length(mean)
                diff_j = x_points[j, idx] - mean[j]
                temp_sum += covar_inv[i, j] * diff_j
            end
            
            diff_sum += diff_i * temp_sum
        end
        

        normalization = (2 * T(π))^nmodes * sqrt(det_covar)
        exponent = -0.5 * diff_sum
        
        output[idx] = exp(exponent) / normalization
    end
    
    return nothing
end

"""
    wignerchar_kernel!(output, xi_points, mean, covar, omega, nmodes, hbar)

CUDA kernel for computing Wigner characteristic function values at multiple points.
"""
function wignerchar_kernel!(output::CuDeviceVector{Complex{T}}, xi_points::CuDeviceMatrix{T},
                                 precomputed_quadratic::CuDeviceMatrix{T}, precomputed_linear::CuDeviceVector{T}) where T
    
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx <= size(xi_points, 2)
        quadratic_term = zero(T)
        for i in 1:size(xi_points, 1)
            for j in 1:size(xi_points, 1)
                quadratic_term += xi_points[i, idx] * precomputed_quadratic[i, j] * xi_points[j, idx]
            end
        end
        
        linear_term = zero(T)
        for i in 1:size(xi_points, 1)
            linear_term += precomputed_linear[i] * xi_points[i, idx]
        end
        

        real_part = -0.5 * quadratic_term
        imag_part = -linear_term
        
        output[idx] = Complex{T}(cos(imag_part), sin(imag_part)) * exp(real_part)
    end
    
    return nothing
end

function Gabs.wigner(state::GaussianState{B,M,V}, x::AbstractVector) where {B<:SymplecticBasis, M<:CuArray, V<:CuArray}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        cpu_state = GaussianState(state.basis, Array(state.mean), Array(state.covar); ħ = state.ħ)
        return Gabs.wigner(cpu_state, x)
    end
    
    basis = state.basis
    nmodes = basis.nmodes
    mean = state.mean
    covar = state.covar
    
    length(mean) == length(x) || throw(ArgumentError(WIGNER_ERROR))
    
    T = eltype(mean)
    x_gpu = CuArray(T.(x))
    
    diff = x_gpu .- mean
    covar_inv = inv(covar)
    
    quad_form = dot(diff, covar_inv * diff)
    
    det_covar = det(covar)
    normalization = (2 * T(π))^nmodes * sqrt(det_covar)
    
    result = exp(-0.5 * quad_form) / normalization
    
    return result  
end

function Gabs.wigner(state::GaussianState{B,M,V}, x_points::CuMatrix{T}) where {B<:SymplecticBasis, M<:CuArray, V<:CuArray, T}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        cpu_state = GaussianState(state.basis, Array(state.mean), Array(state.covar); ħ = state.ħ)
        return [Gabs.wigner(cpu_state, Array(x_points[:, i])) for i in 1:size(x_points, 2)]
    end
    basis = state.basis
    nmodes = basis.nmodes
    size(x_points, 1) == size(state.mean, 1) || throw(ArgumentError(WIGNER_ERROR))
    num_points = size(x_points, 2)
    mean_cpu = Array(state.mean)
    covar_cpu = Array(state.covar)
    covar_inv_cpu = inv(covar_cpu)
    det_covar = det(covar_cpu)
    mean_gpu = CuArray{T}(mean_cpu)
    covar_inv_gpu = CuArray{T}(covar_inv_cpu)
    x_points_gpu = CuArray{T}(x_points)
    output = CUDA.zeros(T, num_points)
    threads_per_block = min(256, num_points)
    num_blocks = cld(num_points, threads_per_block)
    @cuda threads=threads_per_block blocks=num_blocks wigner_kernel!(
        output, x_points_gpu, mean_gpu, covar_inv_gpu, T(det_covar), nmodes
    )
    return output
end

function Gabs.wignerchar(state::GaussianState{B,M,V}, xi::AbstractVector) where {B<:SymplecticBasis, M<:CuArray, V<:CuArray}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        cpu_state = GaussianState(state.basis, Array(state.mean), Array(state.covar); ħ = state.ħ)
        return Gabs.wignerchar(cpu_state, xi)
    end
    basis = state.basis
    nmodes = basis.nmodes
    mean = state.mean
    covar = state.covar
    length(mean) == length(xi) || throw(ArgumentError(WIGNER_ERROR))
    T = eltype(mean)
    xi_gpu = CuArray(T.(xi))
    omega = CuArray(symplecticform(basis))
    temp1 = omega * covar
    temp2 = temp1 * transpose(omega)
    quadratic_term = dot(xi_gpu, temp2 * xi_gpu)
    omega_mean = omega * mean
    linear_term = dot(omega_mean, xi_gpu)
    real_part = -0.5 * quadratic_term  
    imag_part = -linear_term
    result = Complex{T}(cos(imag_part), sin(imag_part)) * exp(real_part)
    return result  
end

function Gabs.wignerchar(state::GaussianState{B,M,V}, xi_points::CuMatrix{T}) where {B<:SymplecticBasis, M<:CuArray, V<:CuArray, T}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        cpu_state = GaussianState(state.basis, Array(state.mean), Array(state.covar); ħ = state.ħ)
        return [Gabs.wignerchar(cpu_state, Array(xi_points[:, i])) for i in 1:size(xi_points, 2)]
    end
    basis = state.basis
    nmodes = basis.nmodes
    size(xi_points, 1) == size(state.mean, 1) || throw(ArgumentError(WIGNER_ERROR))
    num_points = size(xi_points, 2)
    mean_cpu = Array(state.mean)
    covar_cpu = Array(state.covar)
    omega_cpu = symplecticform(basis)
    temp_matrix = omega_cpu * covar_cpu * transpose(omega_cpu)
    precomputed_linear = omega_cpu * mean_cpu
    precomputed_quadratic_gpu = CuArray{T}(temp_matrix)
    precomputed_linear_gpu = CuArray{T}(precomputed_linear)
    xi_points_gpu = CuArray{T}(xi_points)
    output = CUDA.zeros(Complex{T}, num_points)
    threads_per_block = min(256, num_points)
    num_blocks = cld(num_points, threads_per_block)
    @cuda threads=threads_per_block blocks=num_blocks wignerchar_kernel!(
        output, xi_points_gpu, precomputed_quadratic_gpu, precomputed_linear_gpu
    )
    return output
end

"""
    wigner_grid(state::GaussianState, x_range, p_range, nx::Int, np::Int)

Create a phase space grid for Wigner function evaluation.
Returns points as a 2×(nx*np) CuMatrix for efficient GPU evaluation.
"""
function wigner_grid(state::GaussianState{B,M,V}, x_range, p_range, nx::Int, np::Int) where {B<:SymplecticBasis, M<:CuArray, V<:CuArray}
    T = eltype(M)
    x_vals = range(x_range[1], x_range[2], length=nx)
    p_vals = range(p_range[1], p_range[2], length=np)
    total_points = nx * np
    points = CUDA.zeros(T, 2, total_points)
    for i in 1:nx
        for j in 1:np
            idx = (i-1) * np + j
            points[1, idx] = T(x_vals[i])
            points[2, idx] = T(p_vals[j])
        end
    end
    return points
end

"""
    cross_wigner_batch(state1::GaussianState{CuArray}, state2::GaussianState{CuArray}, 
                       x_points::CuMatrix)

 Batch evaluation of cross-Wigner function using existing GPU kernels.
"""
function cross_wigner_batch(state1::GaussianState{B,M,V}, state2::GaussianState{B,M,V}, 
                           x_points::CuMatrix{T}) where {B<:SymplecticBasis, M<:CuArray, V<:CuArray, T}
    if !CUDA_AVAILABLE
        error("GPU cross_wigner_batch called but CUDA not available")
    end
    typeof(state1.basis) == typeof(state2.basis) || throw(ArgumentError(SYMPLECTIC_ERROR))
    state1.basis == state2.basis || throw(ArgumentError(SYMPLECTIC_ERROR))
    state1.ħ == state2.ħ || throw(ArgumentError(HBAR_ERROR))
    
    num_points = size(x_points, 2)
    results = CUDA.zeros(Complex{T}, num_points)
    _launch_cross_wigner_batch_kernel!(results, x_points, state1, state2)
    return results
end

"""
    cross_wignerchar_batch(state1::GaussianState{CuArray}, state2::GaussianState{CuArray}, 
                          xi_points::CuMatrix)

Batch evaluation of cross-Wigner characteristic function using GPU kernels.
"""
function cross_wignerchar_batch(state1::GaussianState{B,M,V}, state2::GaussianState{B,M,V}, 
                               xi_points::CuMatrix{T}) where {B<:SymplecticBasis, M<:CuArray, V<:CuArray, T}
    if !CUDA_AVAILABLE
        error("GPU cross_wignerchar_batch called but CUDA not available")
    end
    typeof(state1.basis) == typeof(state2.basis) || throw(ArgumentError(SYMPLECTIC_ERROR))
    state1.basis == state2.basis || throw(ArgumentError(SYMPLECTIC_ERROR))
    state1.ħ == state2.ħ || throw(ArgumentError(HBAR_ERROR))
    num_points = size(xi_points, 2)
    results = CUDA.zeros(Complex{T}, num_points)
    _launch_cross_wignerchar_batch_kernel!(results, xi_points, state1, state2)
    return results
end

"""
    _launch_cross_wigner_batch_kernel!(results, x_points, state1, state2)

Launch GPU kernel for batch cross-Wigner evaluation.
"""
function _launch_cross_wigner_batch_kernel!(results::CuArray{Complex{T}}, 
                                           x_points::CuMatrix{T},
                                           state1::GaussianState, 
                                           state2::GaussianState) where T
    num_points = size(x_points, 2)
    nmodes = state1.basis.nmodes
    mean1_cpu = Array(state1.mean)
    mean2_cpu = Array(state2.mean)
    covar1_cpu = Array(state1.covar)
    covar2_cpu = Array(state2.covar)
    omega_cpu = symplecticform(state1.basis)
    avg_covar_cpu = (covar1_cpu + covar2_cpu) / 2
    avg_covar_inv_cpu = inv(avg_covar_cpu)
    log_det_avg_covar = logdet(avg_covar_cpu)
    mean1_gpu = CuArray{T}(mean1_cpu)
    mean2_gpu = CuArray{T}(mean2_cpu)
    avg_covar_inv_gpu = CuArray{T}(avg_covar_inv_cpu)
    omega_gpu = CuArray{T}(omega_cpu)
    threads_per_block = min(256, num_points)
    num_blocks = cld(num_points, threads_per_block)
    @cuda threads=threads_per_block blocks=num_blocks _cross_wigner_batch_kernel!(
        results, x_points, mean1_gpu, mean2_gpu, avg_covar_inv_gpu, 
        omega_gpu, T(log_det_avg_covar), nmodes, T(state1.ħ)
    )
    CUDA.synchronize()
end

"""
    _launch_cross_wignerchar_batch_kernel!(results, xi_points, state1, state2)

Launch optimized GPU kernel for batch cross-Wigner characteristic function evaluation.
"""
function _launch_cross_wignerchar_batch_kernel!(results::CuArray{Complex{T}}, 
                                               xi_points::CuMatrix{T},
                                               state1::GaussianState, 
                                               state2::GaussianState) where T
    
    num_points = size(xi_points, 2)
    mean1_cpu = Array(state1.mean)
    mean2_cpu = Array(state2.mean)
    covar1_cpu = Array(state1.covar)
    covar2_cpu = Array(state2.covar)
    omega_cpu = symplecticform(state1.basis)
    avg_mean_cpu = (mean1_cpu + mean2_cpu) / 2
    avg_covar_cpu = (covar1_cpu + covar2_cpu) / 2
    temp_matrix = omega_cpu * avg_covar_cpu * transpose(omega_cpu)
    linear_vector = omega_cpu * avg_mean_cpu
    quadratic_matrix_gpu = CuArray{T}(temp_matrix)
    linear_vector_gpu = CuArray{T}(linear_vector)
    threads_per_block = min(256, num_points)
    num_blocks = cld(num_points, threads_per_block)
    @cuda threads=threads_per_block blocks=num_blocks _cross_wignerchar_batch_kernel!(
        results, xi_points, quadratic_matrix_gpu, linear_vector_gpu
    )
    CUDA.synchronize()
end

"""
    _cross_wigner_batch_kernel!(results, x_points, mean1, mean2, avg_covar_inv, 
                               omega, log_det_avg_covar, nmodes, hbar)

 GPU kernel for batch cross-Wigner evaluation.
"""
function _cross_wigner_batch_kernel!(results::CuDeviceArray{Complex{T}}, 
                                    x_points::CuDeviceMatrix{T},
                                    mean1::CuDeviceVector{T}, 
                                    mean2::CuDeviceVector{T},
                                    avg_covar_inv::CuDeviceMatrix{T},
                                    omega::CuDeviceMatrix{T}, 
                                    log_det_avg_covar::T, 
                                    nmodes::Int, hbar::T) where T

    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= size(x_points, 2)
        dim = size(x_points, 1)
        quadratic_form = zero(T)
        for i in 1:dim
            avg_mean_i = (mean1[i] + mean2[i]) * T(0.5)
            dx_i = x_points[i, idx] - avg_mean_i
            for j in 1:dim
                avg_mean_j = (mean1[j] + mean2[j]) * T(0.5)
                dx_j = x_points[j, idx] - avg_mean_j
                quadratic_form += dx_i * avg_covar_inv[i, j] * dx_j
            end
        end
        phase_arg = zero(T)
        for i in 1:dim
            mu_diff_i = mean1[i] - mean2[i]
            avg_mean_i = (mean1[i] + mean2[i]) * T(0.5)
            dx_i = x_points[i, idx] - avg_mean_i
            omega_dx_i = zero(T)
            for j in 1:dim
                avg_mean_j = (mean1[j] + mean2[j]) * T(0.5)
                dx_j = x_points[j, idx] - avg_mean_j
                omega_dx_i += omega[i, j] * dx_j
            end
            phase_arg += mu_diff_i * omega_dx_i
        end
        log_normalization = -T(nmodes) * log(T(2π)) - T(0.5) * log_det_avg_covar
        normalization = exp(log_normalization)
        gaussian_part = exp(-T(0.5) * quadratic_form)
        sin_val, cos_val = sincos(phase_arg / hbar)
        phase_factor = Complex{T}(cos_val, sin_val)
        results[idx] = normalization * gaussian_part * phase_factor
    end
    return nothing
end

"""
    _cross_wignerchar_batch_kernel!(results, xi_points, quadratic_matrix, linear_vector)

 GPU kernel for batch cross-Wigner characteristic function evaluation.
"""
function _cross_wignerchar_batch_kernel!(results::CuDeviceArray{Complex{T}},
                                        xi_points::CuDeviceMatrix{T},
                                        quadratic_matrix::CuDeviceMatrix{T},
                                        linear_vector::CuDeviceVector{T}) where T
    
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= size(xi_points, 2)
        dim = size(xi_points, 1)
        quadratic_term = zero(T)
        for i in 1:dim
            for j in 1:dim
                quadratic_term += xi_points[i, idx] * quadratic_matrix[i, j] * xi_points[j, idx]
            end
        end
        linear_term = zero(T)
        for i in 1:dim
            linear_term += linear_vector[i] * xi_points[i, idx]
        end
        real_part = -T(0.5) * quadratic_term
        imag_part = -linear_term
        sin_val, cos_val = sincos(imag_part)
        results[idx] = Complex{T}(cos_val, sin_val) * exp(real_part)
    end
    return nothing
end