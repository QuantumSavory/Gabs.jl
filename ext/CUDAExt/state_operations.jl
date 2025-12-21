function Gabs.tensor(::Type{Tm}, ::Type{Tc}, state1::GaussianState{B,M1,V1}, state2::GaussianState{B,M2,V2}) where {
    Tm<:CuVector, Tc<:CuMatrix, B<:SymplecticBasis, M1<:CuArray, V1<:CuArray, M2<:CuArray, V2<:CuArray}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        cpu_state1 = GaussianState(state1.basis, Array(state1.mean), Array(state1.covar); ħ=state1.ħ)
        cpu_state2 = GaussianState(state2.basis, Array(state2.mean), Array(state2.covar); ħ=state2.ħ)
        result_cpu = tensor(Vector{eltype(Tm)}, Matrix{eltype(Tc)}, cpu_state1, cpu_state2)
        return GaussianState(result_cpu.basis, CuArray(result_cpu.mean), CuArray(result_cpu.covar); ħ=result_cpu.ħ)
    end
    typeof(state1.basis) == typeof(state2.basis) || throw(ArgumentError(Gabs.SYMPLECTIC_ERROR))
    state1.ħ == state2.ħ || throw(ArgumentError(Gabs.HBAR_ERROR))
    combined_basis = state1.basis ⊕ state2.basis
    T = eltype(Tm)
    mean_combined = vcat(state1.mean, state2.mean)
    n1 = length(state1.mean)
    n2 = length(state2.mean)
    total_dim = n1 + n2
    covar_combined = CUDA.zeros(T, total_dim, total_dim)
    covar_combined[1:n1, 1:n1] .= state1.covar
    covar_combined[n1+1:end, n1+1:end] .= state2.covar
    return GaussianState(combined_basis, mean_combined, covar_combined; ħ = state1.ħ)
end

function Gabs.tensor(state1::GaussianState{B,<:Array,<:Array}, state2::GaussianState{B,<:CuArray,<:CuArray}) where {B<:SymplecticBasis}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        return tensor(state1, GaussianState(state2.basis, Array(state2.mean), Array(state2.covar); ħ=state2.ħ))
    end
    typeof(state1.basis) == typeof(state2.basis) || throw(ArgumentError(Gabs.SYMPLECTIC_ERROR))
    state1.ħ == state2.ħ || throw(ArgumentError(Gabs.HBAR_ERROR))
    T = real(eltype(state2.mean))
    gpu_state1 = GaussianState(state1.basis, CuArray{T}(state1.mean), CuArray{T}(state1.covar); ħ = state1.ħ)
    return tensor(gpu_state1, state2)
end

function Gabs.tensor(state1::GaussianState{B,<:CuArray,<:CuArray}, state2::GaussianState{B,<:Array,<:Array}) where {B<:SymplecticBasis}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        return tensor(GaussianState(state1.basis, Array(state1.mean), Array(state1.covar); ħ=state1.ħ), state2)
    end
    typeof(state1.basis) == typeof(state2.basis) || throw(ArgumentError(Gabs.SYMPLECTIC_ERROR))
    state1.ħ == state2.ħ || throw(ArgumentError(Gabs.HBAR_ERROR))
    T = real(eltype(state1.mean))
    gpu_state2 = GaussianState(state2.basis, CuArray{T}(state2.mean), CuArray{T}(state2.covar); ħ = state2.ħ)
    return tensor(state1, gpu_state2)
end