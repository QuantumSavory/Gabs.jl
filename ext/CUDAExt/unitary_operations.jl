function Gabs.tensor(::Type{Td}, ::Type{Ts}, op1::GaussianUnitary{B,D1,S1}, op2::GaussianUnitary{B,D2,S2}) where {
    Td<:CuVector, Ts<:CuMatrix, B<:SymplecticBasis, D1<:CuArray, S1<:CuArray, D2<:CuArray, S2<:CuArray}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        cpu_op1 = GaussianUnitary(op1.basis, Array(op1.disp), Array(op1.symplectic); ħ=op1.ħ)
        cpu_op2 = GaussianUnitary(op2.basis, Array(op2.disp), Array(op2.symplectic); ħ=op2.ħ)
        result_cpu = tensor(Vector{eltype(Td)}, Matrix{eltype(Ts)}, cpu_op1, cpu_op2)
        return GaussianUnitary(result_cpu.basis, CuArray{eltype(Td)}(result_cpu.disp), CuArray{eltype(Ts)}(result_cpu.symplectic); ħ=result_cpu.ħ)
    end
    typeof(op1.basis) == typeof(op2.basis) || throw(ArgumentError(Gabs.SYMPLECTIC_ERROR))
    op1.ħ == op2.ħ || throw(ArgumentError(Gabs.HBAR_ERROR))
    combined_basis = op1.basis ⊕ op2.basis
    T = eltype(Td)
    n1 = length(op1.disp)
    n2 = length(op2.disp)
    total_dim = n1 + n2
    disp_combined = vcat(CuArray{T}(op1.disp), CuArray{T}(op2.disp))
    symplectic_combined = CUDA.zeros(T, total_dim, total_dim)
    symplectic_combined[1:n1, 1:n1] .= CuArray{T}(op1.symplectic)
    symplectic_combined[n1+1:end, n1+1:end] .= CuArray{T}(op2.symplectic)
    return GaussianUnitary(combined_basis, disp_combined, symplectic_combined; ħ = op1.ħ)
end

function Gabs.tensor(::Type{Td}, ::Type{Tt}, op1::GaussianChannel{B,D1,T1}, op2::GaussianChannel{B,D2,T2}) where {
    Td<:CuVector, Tt<:CuMatrix, B<:SymplecticBasis, D1<:CuArray, T1<:CuArray, D2<:CuArray, T2<:CuArray}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        cpu_op1 = GaussianChannel(op1.basis, Array(op1.disp), Array(op1.transform), Array(op1.noise); ħ=op1.ħ)
        cpu_op2 = GaussianChannel(op2.basis, Array(op2.disp), Array(op2.transform), Array(op2.noise); ħ=op2.ħ)
        result_cpu = tensor(Vector{eltype(Td)}, Matrix{eltype(Tt)}, cpu_op1, cpu_op2)
        return GaussianChannel(result_cpu.basis, CuArray{eltype(Td)}(result_cpu.disp), CuArray{eltype(Tt)}(result_cpu.transform), CuArray{eltype(Tt)}(result_cpu.noise); ħ=result_cpu.ħ)
    end
    typeof(op1.basis) == typeof(op2.basis) || throw(ArgumentError(Gabs.SYMPLECTIC_ERROR))
    op1.ħ == op2.ħ || throw(ArgumentError(Gabs.HBAR_ERROR))
    combined_basis = op1.basis ⊕ op2.basis  
    T = eltype(Td)
    n1 = length(op1.disp)
    n2 = length(op2.disp) 
    total_dim = n1 + n2
    disp_combined = vcat(CuArray{T}(op1.disp), CuArray{T}(op2.disp))
    transform_combined = CUDA.zeros(T, total_dim, total_dim)
    transform_combined[1:n1, 1:n1] .= CuArray{T}(op1.transform)
    transform_combined[n1+1:end, n1+1:end] .= CuArray{T}(op2.transform)
    noise_combined = CUDA.zeros(T, total_dim, total_dim)
    noise_combined[1:n1, 1:n1] .= CuArray{T}(op1.noise)
    noise_combined[n1+1:end, n1+1:end] .= CuArray{T}(op2.noise)
    return GaussianChannel(combined_basis, disp_combined, transform_combined, noise_combined; ħ = op1.ħ)
end

function Base.:(*)(op::GaussianUnitary{B,<:CuArray,<:CuArray}, state::GaussianState{B,<:Array,<:Array}) where {B<:SymplecticBasis}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        cpu_op = GaussianUnitary(op.basis, Array(op.disp), Array(op.symplectic); ħ=op.ħ)
        return cpu_op * state
    end
    op.basis == state.basis || throw(ArgumentError(ACTION_ERROR))
    op.ħ == state.ħ || throw(ArgumentError(HBAR_ERROR))
    T = real(eltype(op.disp))
    gpu_state = GaussianState(state.basis, CuArray{T}(state.mean), CuArray{T}(state.covar); ħ = state.ħ)
    return op * gpu_state
end

function Base.:(*)(op::GaussianUnitary{B,<:Array,<:Array}, state::GaussianState{B,<:CuArray,<:CuArray}) where {B<:SymplecticBasis}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        cpu_state = GaussianState(state.basis, Array(state.mean), Array(state.covar); ħ=state.ħ)
        return op * cpu_state
    end
    op.basis == state.basis || throw(ArgumentError(ACTION_ERROR))
    op.ħ == state.ħ || throw(ArgumentError(HBAR_ERROR))
    T = real(eltype(state.mean))
    gpu_op = GaussianUnitary(op.basis, CuArray{T}(op.disp), CuArray{T}(op.symplectic); ħ = op.ħ)
    return gpu_op * state
end

function Base.:(*)(op::GaussianChannel{B,<:CuArray,<:CuArray}, state::GaussianState{B,<:Array,<:Array}) where {B<:SymplecticBasis}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        cpu_op = GaussianChannel(op.basis, Array(op.disp), Array(op.transform), Array(op.noise); ħ=op.ħ)
        return cpu_op * state
    end
    op.basis == state.basis || throw(ArgumentError(ACTION_ERROR))
    op.ħ == state.ħ || throw(ArgumentError(HBAR_ERROR))
    T = real(eltype(op.disp))
    gpu_state = GaussianState(state.basis, CuArray{T}(state.mean), CuArray{T}(state.covar); ħ = state.ħ)
    return op * gpu_state
end

function Base.:(*)(op::GaussianChannel{B,<:Array,<:Array}, state::GaussianState{B,<:CuArray,<:CuArray}) where {B<:SymplecticBasis}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        cpu_state = GaussianState(state.basis, Array(state.mean), Array(state.covar); ħ=state.ħ)
        return op * cpu_state
    end
    op.basis == state.basis || throw(ArgumentError(ACTION_ERROR))
    op.ħ == state.ħ || throw(ArgumentError(HBAR_ERROR))
    T = real(eltype(state.mean))
    gpu_op = GaussianChannel(op.basis, CuArray{T}(op.disp), CuArray{T}(op.transform), CuArray{T}(op.noise); ħ = op.ħ)
    return gpu_op * state
end