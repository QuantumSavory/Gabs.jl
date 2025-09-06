"""
    _create_pair_to_block_permutation(nmodes::Int, ::Type{T}) where T

Create GPU permutation indices for QuadPairBasis -> QuadBlockBasis conversion.
QuadPair: [x1,p1,x2,p2,...] -> QuadBlock: [x1,x2,...,p1,p2,...]
"""
function _create_pair_to_block_permutation(nmodes::Int, ::Type{T}) where T
    x_indices = collect(1:2:2*nmodes)  
    p_indices = collect(2:2:2*nmodes) 
    perm_indices = vcat(x_indices, p_indices)  
    return CuArray{Int}(perm_indices)
end

"""
    _create_block_to_pair_permutation(nmodes::Int, ::Type{T}) where T

Create GPU permutation indices for QuadBlockBasis -> QuadPairBasis conversion.
QuadBlock: [x1,x2,...,p1,p2,...] -> QuadPair: [x1,p1,x2,p2,...]
"""
function _create_block_to_pair_permutation(nmodes::Int, ::Type{T}) where T
    perm_indices = Vector{Int}(undef, 2*nmodes)
    for i in 1:nmodes
        perm_indices[2*i-1] = i       
        perm_indices[2*i] = i + nmodes 
    end
    return CuArray{Int}(perm_indices)
end

"""
    _convert_mean_pair_to_block!(mean_out::CuVector{T}, mean_in::CuVector{T}, nmodes::Int) where T

Convert mean vector from QuadPairBasis to QuadBlockBasis using vectorized GPU operations.
"""
function _convert_mean_pair_to_block!(mean_out::CuVector{T}, mean_in::CuVector{T}, nmodes::Int) where T
    x_view = @view mean_in[1:2:2*nmodes]  
    p_view = @view mean_in[2:2:2*nmodes]  
    @views mean_out[1:nmodes] .= x_view
    @views mean_out[nmodes+1:2*nmodes] .= p_view
    return mean_out
end

"""
    _convert_mean_block_to_pair!(mean_out::CuVector{T}, mean_in::CuVector{T}, nmodes::Int) where T

Convert mean vector from QuadBlockBasis to QuadPairBasis using vectorized GPU operations.
"""
function _convert_mean_block_to_pair!(mean_out::CuVector{T}, mean_in::CuVector{T}, nmodes::Int) where T
    x_view = @view mean_in[1:nmodes]           
    p_view = @view mean_in[nmodes+1:2*nmodes]   
    @views mean_out[1:2:2*nmodes] .= x_view     
    @views mean_out[2:2:2*nmodes] .= p_view    
    return mean_out
end

"""
    _convert_matrix_with_permutation(matrix::CuMatrix{T}, perm_indices::CuVector{Int}) where T

Apply permutation to both rows and columns of a matrix using GPU operations.
"""
function _convert_matrix_with_permutation(matrix::CuMatrix{T}, perm_indices::CuVector{Int}) where T
    return matrix[perm_indices, perm_indices]
end

"""
    changebasis(::Type{QuadBlockBasis}, state::GaussianState{QuadPairBasis, CuArray, CuArray})

Convert GaussianState from QuadPairBasis to QuadBlockBasis on GPU.
"""
function changebasis(::Type{B1}, state::GaussianState{B2,M,V}) where {B1<:QuadBlockBasis,B2<:QuadPairBasis,M<:CuArray,V<:CuArray}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        cpu_state = GaussianState(state.basis, Array(state.mean), Array(state.covar); ħ=state.ħ)
        cpu_result = changebasis(B1, cpu_state)
        return GaussianState(cpu_result.basis, CuArray(cpu_result.mean), CuArray(cpu_result.covar); ħ=cpu_result.ħ)
    end
    
    nmodes = state.basis.nmodes
    T = eltype(M)
    mean_new = CUDA.zeros(T, 2*nmodes)
    covar_new = CUDA.zeros(T, 2*nmodes, 2*nmodes)
    _convert_mean_pair_to_block!(mean_new, state.mean, nmodes)
    perm_indices = _create_pair_to_block_permutation(nmodes, T)
    covar_new = _convert_matrix_with_permutation(state.covar, perm_indices)
    return GaussianState(B1(nmodes), mean_new, covar_new; ħ = state.ħ)
end

"""
    changebasis(::Type{QuadPairBasis}, state::GaussianState{QuadBlockBasis, CuArray, CuArray})

Convert GaussianState from QuadBlockBasis to QuadPairBasis on GPU.
"""
function changebasis(::Type{B1}, state::GaussianState{B2,M,V}) where {B1<:QuadPairBasis,B2<:QuadBlockBasis,M<:CuArray,V<:CuArray}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        cpu_state = GaussianState(state.basis, Array(state.mean), Array(state.covar); ħ=state.ħ)
        cpu_result = changebasis(B1, cpu_state)
        return GaussianState(cpu_result.basis, CuArray(cpu_result.mean), CuArray(cpu_result.covar); ħ=cpu_result.ħ)
    end
    nmodes = state.basis.nmodes
    T = eltype(M)
    mean_new = CUDA.zeros(T, 2*nmodes)
    covar_new = CUDA.zeros(T, 2*nmodes, 2*nmodes)
    _convert_mean_block_to_pair!(mean_new, state.mean, nmodes)
    perm_indices = _create_block_to_pair_permutation(nmodes, T)
    covar_new = _convert_matrix_with_permutation(state.covar, perm_indices)
    return GaussianState(B1(nmodes), mean_new, covar_new; ħ = state.ħ)
end

changebasis(::Type{<:QuadBlockBasis}, state::GaussianState{<:QuadBlockBasis,<:CuArray,<:CuArray}) = state
changebasis(::Type{<:QuadPairBasis}, state::GaussianState{<:QuadPairBasis,<:CuArray,<:CuArray}) = state

"""
    changebasis(::Type{QuadBlockBasis}, op::GaussianUnitary{QuadPairBasis, CuArray, CuArray})

Convert GaussianUnitary from QuadPairBasis to QuadBlockBasis on GPU.
"""
function changebasis(::Type{B1}, op::GaussianUnitary{B2,D,S}) where {B1<:QuadBlockBasis,B2<:QuadPairBasis,D<:CuArray,S<:CuArray}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        cpu_op = GaussianUnitary(op.basis, Array(op.disp), Array(op.symplectic); ħ=op.ħ)
        cpu_result = changebasis(B1, cpu_op)
        return GaussianUnitary(cpu_result.basis, CuArray(cpu_result.disp), CuArray(cpu_result.symplectic); ħ=cpu_result.ħ)
    end
    
    nmodes = op.basis.nmodes
    T = eltype(D)
    disp_new = CUDA.zeros(T, 2*nmodes)
    symp_new = CUDA.zeros(T, 2*nmodes, 2*nmodes)
    _convert_mean_pair_to_block!(disp_new, op.disp, nmodes)
    perm_indices = _create_pair_to_block_permutation(nmodes, T)
    symp_new = _convert_matrix_with_permutation(op.symplectic, perm_indices)
    return GaussianUnitary(B1(nmodes), disp_new, symp_new; ħ = op.ħ)
end

"""
    changebasis(::Type{QuadPairBasis}, op::GaussianUnitary{QuadBlockBasis, CuArray, CuArray})

Convert GaussianUnitary from QuadBlockBasis to QuadPairBasis on GPU.
"""
function changebasis(::Type{B1}, op::GaussianUnitary{B2,D,S}) where {B1<:QuadPairBasis,B2<:QuadBlockBasis,D<:CuArray,S<:CuArray}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        cpu_op = GaussianUnitary(op.basis, Array(op.disp), Array(op.symplectic); ħ=op.ħ)
        cpu_result = changebasis(B1, cpu_op)
        return GaussianUnitary(cpu_result.basis, CuArray(cpu_result.disp), CuArray(cpu_result.symplectic); ħ=cpu_result.ħ)
    end
    nmodes = op.basis.nmodes
    T = eltype(D)
    disp_new = CUDA.zeros(T, 2*nmodes)
    symp_new = CUDA.zeros(T, 2*nmodes, 2*nmodes)
    _convert_mean_block_to_pair!(disp_new, op.disp, nmodes)
    perm_indices = _create_block_to_pair_permutation(nmodes, T)
    symp_new = _convert_matrix_with_permutation(op.symplectic, perm_indices)
    return GaussianUnitary(B1(nmodes), disp_new, symp_new; ħ = op.ħ)
end

changebasis(::Type{<:QuadBlockBasis}, op::GaussianUnitary{<:QuadBlockBasis,<:CuArray,<:CuArray}) = op
changebasis(::Type{<:QuadPairBasis}, op::GaussianUnitary{<:QuadPairBasis,<:CuArray,<:CuArray}) = op

"""
    changebasis(::Type{QuadBlockBasis}, op::GaussianChannel{QuadPairBasis, CuArray, CuArray})

Convert GaussianChannel from QuadPairBasis to QuadBlockBasis on GPU.
"""
function changebasis(::Type{B1}, op::GaussianChannel{B2,D,T}) where {B1<:QuadBlockBasis,B2<:QuadPairBasis,D<:CuArray,T<:CuArray}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        cpu_op = GaussianChannel(op.basis, Array(op.disp), Array(op.transform), Array(op.noise); ħ=op.ħ)
        cpu_result = changebasis(B1, cpu_op)
        return GaussianChannel(cpu_result.basis, CuArray(cpu_result.disp), 
                              CuArray(cpu_result.transform), CuArray(cpu_result.noise); ħ=cpu_result.ħ)
    end
    nmodes = op.basis.nmodes
    Td = eltype(D)
    disp_new = CUDA.zeros(Td, 2*nmodes)
    transform_new = CUDA.zeros(Td, 2*nmodes, 2*nmodes)
    noise_new = CUDA.zeros(Td, 2*nmodes, 2*nmodes)
    _convert_mean_pair_to_block!(disp_new, op.disp, nmodes)
    perm_indices = _create_pair_to_block_permutation(nmodes, Td)
    transform_new = _convert_matrix_with_permutation(op.transform, perm_indices)
    noise_new = _convert_matrix_with_permutation(op.noise, perm_indices)
    return GaussianChannel(B1(nmodes), disp_new, transform_new, noise_new; ħ = op.ħ)
end

"""
    changebasis(::Type{QuadPairBasis}, op::GaussianChannel{QuadBlockBasis, CuArray, CuArray})

Convert GaussianChannel from QuadBlockBasis to QuadPairBasis on GPU.
"""
function changebasis(::Type{B1}, op::GaussianChannel{B2,D,T}) where {B1<:QuadPairBasis,B2<:QuadBlockBasis,D<:CuArray,T<:CuArray}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        cpu_op = GaussianChannel(op.basis, Array(op.disp), Array(op.transform), Array(op.noise); ħ=op.ħ)
        cpu_result = changebasis(B1, cpu_op)
        return GaussianChannel(cpu_result.basis, CuArray(cpu_result.disp),
                              CuArray(cpu_result.transform), CuArray(cpu_result.noise); ħ=cpu_result.ħ)
    end
    nmodes = op.basis.nmodes
    Td = eltype(D)
    disp_new = CUDA.zeros(Td, 2*nmodes)
    transform_new = CUDA.zeros(Td, 2*nmodes, 2*nmodes)
    noise_new = CUDA.zeros(Td, 2*nmodes, 2*nmodes)
    _convert_mean_block_to_pair!(disp_new, op.disp, nmodes)
    perm_indices = _create_block_to_pair_permutation(nmodes, Td)
    transform_new = _convert_matrix_with_permutation(op.transform, perm_indices)
    noise_new = _convert_matrix_with_permutation(op.noise, perm_indices)
    return GaussianChannel(B1(nmodes), disp_new, transform_new, noise_new; ħ = op.ħ)
end
changebasis(::Type{<:QuadBlockBasis}, op::GaussianChannel{<:QuadBlockBasis,<:CuArray,<:CuArray}) = op
changebasis(::Type{<:QuadPairBasis}, op::GaussianChannel{<:QuadPairBasis,<:CuArray,<:CuArray}) = op