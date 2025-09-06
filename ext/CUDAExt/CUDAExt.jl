module CUDAExt
using CUDA
using CUDA: CuArray, CuVector, CuMatrix, @cuda, threadIdx, blockIdx, blockDim, gridDim
using Gabs
using Gabs: SymplecticBasis, QuadPairBasis, QuadBlockBasis, GaussianState, GaussianUnitary, GaussianChannel, GaussianLinearCombination, _promote_output_vector, _promote_output_matrix, _vacuumstate, _coherentstate, _squeezedstate, _thermalstate, _eprstate, _displace, _squeeze, _twosqueeze, _phaseshift, _beamsplitter, _attenuator, _amplifier, symplecticform, WIGNER_ERROR, ACTION_ERROR, HBAR_ERROR, INDEX_ERROR, SYMPLECTIC_ERROR, cross_wigner, cross_wignerchar, vacuumstate, randstate, randunitary, randchannel, randsymplectic
using LinearAlgebra
using LinearAlgebra: I, det, mul!, eigvals, Diagonal, logdet, dot, inv, diag, cholesky, Symmetric, qr, Hermitian
using Random
using Random: randn!
import Gabs: purity, entropy_vn, vacuumstate, changebasis, randstate, randunitary, randchannel, randsymplectic, ptrace, cross_wigner, cross_wignerchar, wigner, wignerchar, catstate_even, catstate_odd, catstate, gkpstate, _promote_output_vector, _promote_output_matrix, _is_gpu_array, _detect_device, coherentstate, squeezedstate, thermalstate, eprstate, displace, squeeze, twosqueeze, phaseshift, beamsplitter, attenuator, amplifier


const CUDA_AVAILABLE = CUDA.functional()

function __init__()
    if CUDA_AVAILABLE
        @info "CUDA.jl extension loaded successfully. GPU acceleration enabled."
    else
        @warn "CUDA not available. GPU operations will fall back to CPU."
    end
end

function Gabs._gpu_impl(state::GaussianState, precision)
    gpu_mean = CuArray{precision}(state.mean)
    gpu_covar = CuArray{precision}(state.covar)
    return GaussianState(state.basis, gpu_mean, gpu_covar; ħ = state.ħ)
end

function Gabs._gpu_impl(op::GaussianUnitary, precision)
    gpu_disp = CuArray{precision}(op.disp)
    gpu_symplectic = CuArray{precision}(op.symplectic)
    return GaussianUnitary(op.basis, gpu_disp, gpu_symplectic; ħ = op.ħ)
end

function Gabs._gpu_impl(op::GaussianChannel, precision)
    gpu_disp = CuArray{precision}(op.disp)
    gpu_transform = CuArray{precision}(op.transform)
    gpu_noise = CuArray{precision}(op.noise)
    return GaussianChannel(op.basis, gpu_disp, gpu_transform, gpu_noise; ħ = op.ħ)
end

function Gabs._gpu_impl(lc::GaussianLinearCombination, precision)
    cpu_coeffs = Vector{precision}(Array(lc.coeffs))
    gpu_states = [Gabs._gpu_impl(state, precision) for state in lc.states]
    return GaussianLinearCombination(lc.basis, cpu_coeffs, gpu_states)
end

function Gabs._gpu_impl(x::AbstractArray, precision)
    return CuArray{precision}(x)
end

function Gabs._detect_device(x::CuArray)
    return :gpu
end

function Gabs._randstate_gpu_impl(basis::SymplecticBasis, precision; pure=false, ħ=2)
    cpu_state = randstate(basis; pure=pure, ħ=ħ)
    return Gabs._gpu_impl(cpu_state, precision)
end

function Gabs._randunitary_gpu_impl(basis::SymplecticBasis, precision; passive=false, ħ=2)
    cpu_unitary = randunitary(basis; passive=passive, ħ=ħ)
    return Gabs._gpu_impl(cpu_unitary, precision)
end

function Gabs._randchannel_gpu_impl(basis::SymplecticBasis, precision; ħ=2)
    cpu_channel = randchannel(basis; ħ=ħ)
    return Gabs._gpu_impl(cpu_channel, precision)
end

function Gabs._randsymplectic_gpu_impl(basis::SymplecticBasis, precision; passive=false)
    cpu_symplectic = randsymplectic(basis; passive=passive)
    return CuArray{precision}(cpu_symplectic)
end

function Gabs._batch_randstate_gpu_impl(basis::SymplecticBasis, n::Int, precision; pure=false, ħ=2)
    cpu_states = [randstate(basis; pure=pure, ħ=ħ) for _ in 1:n]
    return [Gabs._gpu_impl(state, precision) for state in cpu_states]
end

function Gabs._batch_randunitary_gpu_impl(basis::SymplecticBasis, n::Int, precision; passive=false, ħ=2)
    cpu_unitaries = [randunitary(basis; passive=passive, ħ=ħ) for _ in 1:n]
    return [Gabs._gpu_impl(unitary, precision) for unitary in cpu_unitaries]
end

function _extract_gpu_scalar(x::CuArray{T}) where T
    if ndims(x) == 0
        return T(Array(x)[])
    elseif length(x) == 1
        return T(Array(x)[1])
    else
        return Vector{T}(Array(x))
    end
end

function Gabs.coherentstate(basis::SymplecticBasis{N}, alpha::CuArray; ħ=2) where {N<:Int}
    T = real(eltype(alpha))
    alpha_val = _extract_gpu_scalar(alpha)
    return coherentstate(CuVector{T}, CuMatrix{T}, basis, alpha_val; ħ=ħ)
end

function Gabs.squeezedstate(basis::SymplecticBasis{N}, r::CuArray, theta; ħ=2) where {N<:Int}
    T = eltype(r)
    r_val = _extract_gpu_scalar(r)
    return squeezedstate(CuVector{T}, CuMatrix{T}, basis, r_val, theta; ħ=ħ)
end

function Gabs.thermalstate(basis::SymplecticBasis{N}, photons::CuArray; ħ=2) where {N<:Int}
    T = eltype(photons)
    photons_val = _extract_gpu_scalar(photons)
    return thermalstate(CuVector{T}, CuMatrix{T}, basis, photons_val; ħ=ħ)
end

function Gabs.eprstate(basis::SymplecticBasis{N}, r::CuArray, theta; ħ=2) where {N<:Int}
    T = eltype(r)
    r_val = _extract_gpu_scalar(r)
    return eprstate(CuVector{T}, CuMatrix{T}, basis, r_val, theta; ħ=ħ)
end

function Gabs.displace(basis::SymplecticBasis{N}, alpha::CuArray; ħ=2) where {N<:Int}
    T = real(eltype(alpha))
    alpha_val = _extract_gpu_scalar(alpha)
    return displace(CuVector{T}, CuMatrix{T}, basis, alpha_val; ħ=ħ)
end

function Gabs.squeeze(basis::SymplecticBasis{N}, r::CuArray, theta; ħ=2) where {N<:Int}
    T = eltype(r)
    r_val = _extract_gpu_scalar(r)
    return squeeze(CuVector{T}, CuMatrix{T}, basis, r_val, theta; ħ=ħ)
end

function Gabs.twosqueeze(basis::SymplecticBasis{N}, r::CuArray, theta; ħ=2) where {N<:Int}
    T = eltype(r)
    r_val = _extract_gpu_scalar(r)
    return twosqueeze(CuVector{T}, CuMatrix{T}, basis, r_val, theta; ħ=ħ)
end

function Gabs.phaseshift(basis::SymplecticBasis{N}, theta::CuArray; ħ=2) where {N<:Int}
    T = eltype(theta)
    theta_val = _extract_gpu_scalar(theta)
    return phaseshift(CuVector{T}, CuMatrix{T}, basis, theta_val; ħ=ħ)
end

function Gabs.beamsplitter(basis::SymplecticBasis{N}, transmit::CuArray; ħ=2) where {N<:Int}
    T = eltype(transmit)
    transmit_val = _extract_gpu_scalar(transmit)
    return beamsplitter(CuVector{T}, CuMatrix{T}, basis, transmit_val; ħ=ħ)
end

function Gabs.attenuator(basis::SymplecticBasis{N}, theta::CuArray, n; ħ=2) where {N<:Int}
    T = eltype(theta)
    theta_val = _extract_gpu_scalar(theta)
    return attenuator(CuVector{T}, CuMatrix{T}, basis, theta_val, n; ħ=ħ)
end

function Gabs.amplifier(basis::SymplecticBasis{N}, r::CuArray, n; ħ=2) where {N<:Int}
    T = eltype(r)
    r_val = _extract_gpu_scalar(r)
    return amplifier(CuVector{T}, CuMatrix{T}, basis, r_val, n; ħ=ħ)
end

function Gabs.tensor(state1::GaussianState{B,M1,V1}, state2::GaussianState{B,M2,V2}) where {B<:SymplecticBasis, M1<:CuArray, V1<:CuArray, M2<:CuArray, V2<:CuArray}
    T = promote_type(eltype(M1), eltype(M2))
    return tensor(CuVector{T}, CuMatrix{T}, state1, state2)
end

function Gabs.tensor(::Type{T}, op1::GaussianUnitary{B,D1,S1}, op2::GaussianUnitary{B,D2,S2}) where {
    T<:CuVector, B<:SymplecticBasis, D1<:CuArray, S1<:CuArray, D2<:CuArray, S2<:CuArray}
    return tensor(T, CuMatrix{eltype(T)}, op1, op2)
end

function Gabs.tensor(op1::GaussianUnitary{B,D1,S1}, op2::GaussianUnitary{B,D2,S2}) where {
    B<:SymplecticBasis, D1<:CuArray, S1<:CuArray, D2<:CuArray, S2<:CuArray}
    T = promote_type(eltype(D1), eltype(D2))
    return tensor(CuVector{T}, CuMatrix{T}, op1, op2)
end

function Gabs.tensor(::Type{T}, op1::GaussianChannel{B,D1,T1}, op2::GaussianChannel{B,D2,T2}) where {
    T<:CuVector, B<:SymplecticBasis, D1<:CuArray, T1<:CuArray, D2<:CuArray, T2<:CuArray}
    return tensor(T, CuMatrix{eltype(T)}, op1, op2)
end

function Gabs.tensor(op1::GaussianChannel{B,D1,T1}, op2::GaussianChannel{B,D2,T2}) where {
    B<:SymplecticBasis, D1<:CuArray, T1<:CuArray, D2<:CuArray, T2<:CuArray}
    T = promote_type(eltype(D1), eltype(D2))
    return tensor(CuVector{T}, CuMatrix{T}, op1, op2)
end

function Gabs._adapt_device_gpu(target_constructor, source_obj, args...)
    if source_obj isa GaussianState
        T = real(eltype(source_obj.mean))
    elseif source_obj isa GaussianUnitary
        T = real(eltype(source_obj.disp))
    elseif source_obj isa GaussianChannel
        T = real(eltype(source_obj.disp))
    elseif source_obj isa GaussianLinearCombination && !isempty(source_obj.states)
        T = real(eltype(source_obj.states[1].mean))
    elseif source_obj isa AbstractArray
        T = real(eltype(source_obj))
    else
        T = Float32 
    end
    cpu_obj = target_constructor(args...)
    return Gabs._gpu_impl(cpu_obj, T)
end

function Gabs.isgaussian(state::GaussianState{B,M,V}; atol::R1 = 0, rtol::R2 = atol) where {B<:SymplecticBasis, M<:CuArray, V<:CuArray, R1<:Real, R2<:Real}
    cpu_state = GaussianState(state.basis, Array(state.mean), Array(state.covar); ħ = state.ħ)
    return isgaussian(cpu_state; atol = atol, rtol = rtol)
end

function Gabs.isgaussian(op::GaussianUnitary{B,D,S}; atol::R1 = 0, rtol::R2 = atol) where {B<:SymplecticBasis, D<:CuArray, S<:CuArray, R1<:Real, R2<:Real}
    return issymplectic(op.basis, Array(op.symplectic); atol = atol, rtol = rtol)
end

function Gabs.isgaussian(op::GaussianChannel{B,D,T}; atol::R1 = 0, rtol::R2 = atol) where {B<:SymplecticBasis, D<:CuArray, T<:CuArray, R1<:Real, R2<:Real}
    cpu_op = GaussianChannel(op.basis, Array(op.disp), Array(op.transform), Array(op.noise); ħ = op.ħ)
    return isgaussian(cpu_op; atol = atol, rtol = rtol)
end

function Gabs.issymplectic(basis::SymplecticBasis, x::CuMatrix; atol::R1 = 0, rtol::R2 = atol) where {R1<:Real, R2<:Real}
    return issymplectic(basis, Array(x); atol = atol, rtol = rtol)
end

include("utils.jl")
include("state_operations.jl") 
include("unitary_operations.jl")
include("wigner_kernels.jl")
include("cross_wigner_kernels.jl")
include("interference_wigner.jl")
include("linear_combinations.jl")
include("basis_operations.jl")
include("ptrace_operations.jl")
end