Gabs._is_gpu_array(x::CuArray) = true

function Gabs._detect_device(lc::GaussianLinearCombination)
    if isempty(lc.states)
        return :cpu
    end
    return Gabs._detect_device(lc.states[1].mean)
end

function Gabs._gpu_impl(lc::GaussianLinearCombination, precision::Type{T}) where T
    if !CUDA_AVAILABLE
        @warn "CUDA not available. Cannot transfer to GPU."
        return lc
    end
    gpu_states = [Gabs._gpu_impl(state, precision) for state in lc.states]
    cpu_coeffs = Vector{T}(Array(lc.coeffs))  
    return GaussianLinearCombination(lc.basis, cpu_coeffs, gpu_states)
end

function Base.:(*)(op::GaussianUnitary{B,<:CuArray,<:CuArray}, lc::GaussianLinearCombination{B,C,S}) where {B<:SymplecticBasis,C,S<:GaussianState{B,<:CuArray,<:CuArray}}
    op.basis == lc.basis || throw(ArgumentError(ACTION_ERROR))
    op.ħ == lc.ħ || throw(ArgumentError(HBAR_ERROR))
    new_states = [op * state for state in lc.states]
    return GaussianLinearCombination(lc.basis, Vector(lc.coeffs), new_states)
end

function Base.:(*)(op::GaussianChannel{B,<:CuArray,<:CuArray}, lc::GaussianLinearCombination{B,C,S}) where {B<:SymplecticBasis,C,S<:GaussianState{B,<:CuArray,<:CuArray}}
    op.basis == lc.basis || throw(ArgumentError(ACTION_ERROR))
    op.ħ == lc.ħ || throw(ArgumentError(HBAR_ERROR))
    new_states = [op * state for state in lc.states]
    return GaussianLinearCombination(lc.basis, Vector(lc.coeffs), new_states)
end

function Base.:(*)(op::GaussianUnitary{B,<:Array,<:Array}, lc::GaussianLinearCombination{B,C,S}) where {B<:SymplecticBasis,C,S<:GaussianState{B,<:CuArray,<:CuArray}}
    op.basis == lc.basis || throw(ArgumentError(ACTION_ERROR))
    op.ħ == lc.ħ || throw(ArgumentError(HBAR_ERROR))
    T = real(eltype(lc.states[1].mean))
    gpu_op = Gabs._gpu_impl(op, T)
    new_states = [gpu_op * state for state in lc.states]
    return GaussianLinearCombination(lc.basis, Vector(lc.coeffs), new_states)
end

function Base.:(*)(op::GaussianChannel{B,<:Array,<:Array}, lc::GaussianLinearCombination{B,C,S}) where {B<:SymplecticBasis,C,S<:GaussianState{B,<:CuArray,<:CuArray}}
    op.basis == lc.basis || throw(ArgumentError(ACTION_ERROR))
    op.ħ == lc.ħ || throw(ArgumentError(HBAR_ERROR))
    T = real(eltype(lc.states[1].mean))
    gpu_op = Gabs._gpu_impl(op, T)
    new_states = [gpu_op * state for state in lc.states]
    return GaussianLinearCombination(lc.basis, Vector(lc.coeffs), new_states)
end

function purity(lc::GaussianLinearCombination{B,C,S}) where {B<:SymplecticBasis,C,S<:GaussianState{B,<:CuArray,<:CuArray}}
    return 1.0
end

function entropy_vn(lc::GaussianLinearCombination{B,C,S}) where {B<:SymplecticBasis,C,S<:GaussianState{B,<:CuArray,<:CuArray}}
    return 0.0
end