function vacuumstate(::Type{Tm}, ::Type{Tc}, basis::SymplecticBasis{N}; ħ = 2) where {Tm<:CuVector, Tc<:CuMatrix, N<:Int}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        return vacuumstate(Vector{eltype(Tm)}, Matrix{eltype(Tc)}, basis; ħ = ħ)
    end
    nmodes = basis.nmodes
    T = eltype(Tm)
    mean = CUDA.zeros(T, 2*nmodes)
    covar = CuMatrix{T}((T(ħ)/2) * I(2*nmodes))
    return GaussianState(basis, mean, covar; ħ = ħ)
end

function thermalstate(::Type{Tm}, ::Type{Tc}, basis::SymplecticBasis{N}, photons::P; ħ = 2) where {Tm<:CuVector, Tc<:CuMatrix, N<:Int, P}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        return thermalstate(Vector{eltype(Tm)}, Matrix{eltype(Tc)}, basis, photons; ħ = ħ)
    end
    T = eltype(Tm)
    mean, covar = _thermalstate(basis, photons; ħ = ħ)
    gpu_mean = CuArray{T}(mean)
    gpu_covar = CuArray{T}(covar)
    return GaussianState(basis, gpu_mean, gpu_covar; ħ = ħ)
end

function coherentstate(::Type{Tm}, ::Type{Tc}, basis::SymplecticBasis{N}, alpha::A; ħ = 2) where {Tm<:CuVector, Tc<:CuMatrix, N<:Int, A}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        return coherentstate(Vector{eltype(Tm)}, Matrix{eltype(Tc)}, basis, alpha; ħ = ħ)
    end
    T = eltype(Tm)
    mean, covar = _coherentstate(basis, alpha; ħ = ħ)
    gpu_mean = CuArray{T}(mean)
    gpu_covar = CuArray{T}(covar)
    return GaussianState(basis, gpu_mean, gpu_covar; ħ = ħ)
end

function squeezedstate(::Type{Tm}, ::Type{Tc}, basis::SymplecticBasis{N}, r::R, theta::R; ħ = 2) where {Tm<:CuVector, Tc<:CuMatrix, N<:Int, R}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        return squeezedstate(Vector{eltype(Tm)}, Matrix{eltype(Tc)}, basis, r, theta; ħ = ħ)
    end
    T = eltype(Tm)
    mean, covar = _squeezedstate(basis, r, theta; ħ = ħ)
    gpu_mean = CuArray{T}(mean)
    gpu_covar = CuArray{T}(covar)
    return GaussianState(basis, gpu_mean, gpu_covar; ħ = ħ)
end

function eprstate(::Type{Tm}, ::Type{Tc}, basis::SymplecticBasis{N}, r::R, theta::R; ħ = 2) where {Tm<:CuVector, Tc<:CuMatrix, N<:Int, R}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        return eprstate(Vector{eltype(Tm)}, Matrix{eltype(Tc)}, basis, r, theta; ħ = ħ)
    end
    T = eltype(Tm)
    mean, covar = _eprstate(basis, r, theta; ħ = ħ)
    gpu_mean = CuArray{T}(mean)
    gpu_covar = CuArray{T}(covar)
    
    return GaussianState(basis, gpu_mean, gpu_covar; ħ = ħ)
end

function vacuumstate(::Type{T}, basis::SymplecticBasis{N}; ħ = 2) where {T<:CuVector, N<:Int}
    return vacuumstate(T, CuMatrix{eltype(T)}, basis; ħ = ħ)
end

function coherentstate(::Type{T}, basis::SymplecticBasis{N}, alpha::A; ħ = 2) where {T<:CuVector, N<:Int, A}
    return coherentstate(T, CuMatrix{eltype(T)}, basis, alpha; ħ = ħ)
end

function squeezedstate(::Type{T}, basis::SymplecticBasis{N}, r::R, theta::R; ħ = 2) where {T<:CuVector, N<:Int, R}
    return squeezedstate(T, CuMatrix{eltype(T)}, basis, r, theta; ħ = ħ)
end

function thermalstate(::Type{T}, basis::SymplecticBasis{N}, photons::P; ħ = 2) where {T<:CuVector, N<:Int, P}
    return thermalstate(T, CuMatrix{eltype(T)}, basis, photons; ħ = ħ)
end

function eprstate(::Type{T}, basis::SymplecticBasis{N}, r::R, theta::R; ħ = 2) where {T<:CuVector, N<:Int, R}
    return eprstate(T, CuMatrix{eltype(T)}, basis, r, theta; ħ = ħ)
end
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