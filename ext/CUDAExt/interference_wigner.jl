"""
    wigner(lc::GaussianLinearCombination{GPU states}, x::AbstractVector)

Implements: W(x) = Σᵢ |cᵢ|² Wᵢ(x) + 2 Σᵢ<ⱼ Re(cᵢ*cⱼ W_cross(ψᵢ,ψⱼ,x))
"""
function wigner(lc::GaussianLinearCombination{B,C,S}, x::AbstractVector) where {
    B<:SymplecticBasis, C, S<:GaussianState{B,<:CuArray,<:CuArray}}
    if !CUDA_AVAILABLE
        error("GPU interference wigner called but CUDA not available")
    end
    if !isempty(lc.states)
        expected_len = size(lc.states[1].mean, 1) 
        actual_len = length(x)
        actual_len == expected_len || throw(ArgumentError(WIGNER_ERROR))
    else
        throw(ArgumentError("Cannot compute Wigner function of empty linear combination"))
    end
    T = real(eltype(lc.states[1].mean))
    result = zero(T)    
    x_gpu = CuArray{T}(x)
    n_states = length(lc.states)
    @inbounds for i in 1:n_states
        ci = lc.coeffs[i]  
        state_i = lc.states[i]  
        wigner_i = wigner(state_i, x_gpu)  
        result += abs2(ci) * wigner_i
    end
    @inbounds for i in 1:(n_states-1)
        ci = lc.coeffs[i]  
        state_i = lc.states[i] 
        @inbounds for j in (i+1):n_states
            cj = lc.coeffs[j]  
            state_j = lc.states[j]  
            cross_w = cross_wigner(state_i, state_j, x_gpu) 
            interference_coeff = conj(ci) * cj
            cross_term = 2 * real(interference_coeff * cross_w)
            result += cross_term
        end
    end
    return result
end

"""
    wignerchar(lc::GaussianLinearCombination{GPU states}, xi::AbstractVector)

Implements: χ(ξ) = Σᵢ |cᵢ|² χᵢ(ξ) + 2 Σᵢ<ⱼ Re(cᵢ*cⱼ χ_cross(ψᵢ,ψⱼ,ξ))
"""
function wignerchar(lc::GaussianLinearCombination{B,C,S}, xi::AbstractVector) where {
    B<:SymplecticBasis, C, S<:GaussianState{B,<:CuArray,<:CuArray}}
    if !CUDA_AVAILABLE
        error("GPU interference wignerchar called but CUDA not available")
    end
    if isempty(lc.states)
        throw(ArgumentError("Cannot compute Wigner characteristic function of empty linear combination"))
    end
    
    expected_len = size(lc.states[1].mean, 1)  
    actual_len = length(xi)
    actual_len == expected_len || throw(ArgumentError("Evaluation point dimension ($actual_len) does not match state dimension ($expected_len)"))
    T = real(eltype(lc.states[1].mean))
    result = complex(zero(T), zero(T))
    xi_gpu = CuArray{T}(xi)
    n_states = length(lc.states)
    @inbounds for i in 1:n_states
        ci = lc.coeffs[i] 
        state_i = lc.states[i] 
        wignerchar_i = wignerchar(state_i, xi_gpu) 
        result += abs2(ci) * wignerchar_i
    end
    @inbounds for i in 1:(n_states-1)
        ci = lc.coeffs[i] 
        state_i = lc.states[i] 
        @inbounds for j in (i+1):n_states
            cj = lc.coeffs[j]  
            state_j = lc.states[j]  
            cross_char = cross_wignerchar(state_i, state_j, xi_gpu) 
            interference_coeff = conj(ci) * cj
            cross_term = 2 * real(interference_coeff * cross_char)
            result += cross_term
        end
    end
    return result
end

"""
    wigner(lc::GaussianLinearCombination{GPU states}, x::Vector{T}) where T<:Real
"""
function wigner(lc::GaussianLinearCombination{B,C,S}, x::Vector{T}) where {
    B<:SymplecticBasis, C, S<:GaussianState{B,<:CuArray,<:CuArray}, T<:Real}
    target_precision = real(eltype(lc.states[1].mean))
    return wigner(lc, CuArray{target_precision}(x))
end

"""
    wignerchar(lc::GaussianLinearCombination{GPU states}, xi::Vector{T}) where T<:Real
"""
function wignerchar(lc::GaussianLinearCombination{B,C,S}, xi::Vector{T}) where {
    B<:SymplecticBasis, C, S<:GaussianState{B,<:CuArray,<:CuArray}, T<:Real}
    target_precision = real(eltype(lc.states[1].mean))
    return wignerchar(lc, CuArray{target_precision}(xi))
end

"""
    wigner(lc::GaussianLinearCombination{GPU states}, x_points::CuMatrix{T})

Implements batched version of: W(x) = Σᵢ |cᵢ|² Wᵢ(x) + 2 Σᵢ<ⱼ Re(cᵢ*cⱼ W_cross(ψᵢ,ψⱼ,x))

# Arguments
- `lc::GaussianLinearCombination`: Linear combination with GPU states
- `x_points::CuMatrix{T}`: Phase space points (dimension × num_points)

# Returns  
- `CuArray{T}`: Wigner function values at all points
"""
function wigner(lc::GaussianLinearCombination{B,C,S}, x_points::CuMatrix{T}) where {
    B<:SymplecticBasis, C, S<:GaussianState{B,<:CuArray,<:CuArray}, T}
    if !CUDA_AVAILABLE
        error("GPU batched interference wigner called but CUDA not available")
    end    
    if isempty(lc.states)
        throw(ArgumentError("Cannot compute Wigner function of empty linear combination"))
    end
    expected_len = size(lc.states[1].mean, 1)
    actual_len = size(x_points, 1)
    actual_len == expected_len || throw(ArgumentError("Phase space point dimension ($actual_len) does not match state dimension ($expected_len)"))
    num_points = size(x_points, 2)
    n_states = length(lc.states)
    results = CUDA.zeros(T, num_points)
    for i in 1:n_states
        ci = lc.coeffs[i]  
        state_i = lc.states[i]  
        wigner_values = wigner(state_i, x_points) 
        results .+= abs2(ci) .* wigner_values
    end
    for i in 1:(n_states-1)
        ci = lc.coeffs[i]
        state_i = lc.states[i]
        for j in (i+1):n_states
            cj = lc.coeffs[j]
            state_j = lc.states[j]
            cross_values = cross_wigner_batch(state_i, state_j, x_points)
            interference_coeff = conj(ci) * cj
            cross_contributions = 2 .* real.(interference_coeff .* cross_values)
            results .+= cross_contributions
        end
    end
    return results
end

"""
    wignerchar(lc::GaussianLinearCombination{GPU states}, xi_points::CuMatrix{T})

Implements batched version of: χ(ξ) = Σᵢ |cᵢ|² χᵢ(ξ) + 2 Σᵢ<ⱼ Re(cᵢ*cⱼ χ_cross(ψᵢ,ψⱼ,ξ))

# Arguments
- `lc::GaussianLinearCombination`: Linear combination with GPU states  
- `xi_points::CuMatrix{T}`: Evaluation points in phase space (dimension × num_points)

# Returns
- `CuArray{Complex{T}}`: Characteristic function values at all points
"""
function wignerchar(lc::GaussianLinearCombination{B,C,S}, xi_points::CuMatrix{T}) where {
    B<:SymplecticBasis, C, S<:GaussianState{B,<:CuArray,<:CuArray}, T}
    if !CUDA_AVAILABLE
        error("GPU batched interference wignerchar called but CUDA not available")
    end
    if isempty(lc.states)
        throw(ArgumentError("Cannot compute Wigner characteristic function of empty linear combination"))
    end
    expected_len = size(lc.states[1].mean, 1)
    actual_len = size(xi_points, 1)
    actual_len == expected_len || throw(ArgumentError("Evaluation point dimension ($actual_len) does not match state dimension ($expected_len)"))
    num_points = size(xi_points, 2)
    n_states = length(lc.states)
    results = CUDA.zeros(Complex{T}, num_points)
    for i in 1:n_states
        ci = lc.coeffs[i]
        state_i = lc.states[i]
        wignerchar_values = wignerchar(state_i, xi_points)
        results .+= abs2(ci) .* wignerchar_values
    end
    for i in 1:(n_states-1)
        ci = lc.coeffs[i]
        state_i = lc.states[i]
        for j in (i+1):n_states
            cj = lc.coeffs[j]
            state_j = lc.states[j]
            cross_values = cross_wignerchar_batch(state_i, state_j, xi_points)
            interference_coeff = conj(ci) * cj
            cross_contributions = 2 .* real.(interference_coeff .* cross_values)
            results .+= cross_contributions
        end
    end
    return results
end

"""
    wigner(lc::GaussianLinearCombination{GPU states}, x_points::Matrix{T}) where T<:Real
"""
function wigner(lc::GaussianLinearCombination{B,C,S}, x_points::Matrix{T}) where {
    B<:SymplecticBasis, C, S<:GaussianState{B,<:CuArray,<:CuArray}, T<:Real}
    target_precision = real(eltype(lc.states[1].mean))
    return wigner(lc, CuArray{target_precision}(x_points))
end

"""
    wignerchar(lc::GaussianLinearCombination{GPU states}, xi_points::Matrix{T}) where T<:Real
"""
function wignerchar(lc::GaussianLinearCombination{B,C,S}, xi_points::Matrix{T}) where {
    B<:SymplecticBasis, C, S<:GaussianState{B,<:CuArray,<:CuArray}, T<:Real}
    target_precision = real(eltype(lc.states[1].mean))
    return wignerchar(lc, CuArray{target_precision}(xi_points))
end