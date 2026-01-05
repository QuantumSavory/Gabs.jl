"""
    gpu(x; precision=Float32)

Transfer data to GPU. Works with states, operators, and arrays.
Requires CUDA.jl to be loaded.
"""
function gpu(state::GaussianState; precision=Float32)
    if !isdefined(Base, :get_extension) || isnothing(Base.get_extension(@__MODULE__, :CUDAExt))
        error("GPU functionality requires CUDA.jl. Please run: using CUDA")
    end
    return _gpu_impl(state, precision)
end

function gpu(op::GaussianUnitary; precision=Float32)
    if !isdefined(Base, :get_extension) || isnothing(Base.get_extension(@__MODULE__, :CUDAExt))
        error("GPU functionality requires CUDA.jl. Please run: using CUDA")
    end
    return _gpu_impl(op, precision)
end

function gpu(op::GaussianChannel; precision=Float32)
    if !isdefined(Base, :get_extension) || isnothing(Base.get_extension(@__MODULE__, :CUDAExt))
        error("GPU functionality requires CUDA.jl. Please run: using CUDA")
    end
    return _gpu_impl(op, precision)
end

function gpu(lc::GaussianLinearCombination; precision=Float32)
    if !isdefined(Base, :get_extension) || isnothing(Base.get_extension(@__MODULE__, :CUDAExt))
        error("GPU functionality requires CUDA.jl. Please run: using CUDA")
    end
    return _gpu_impl(lc, precision)
end

function gpu(basis::SymplecticBasis)
    return basis  
end

function gpu(x::AbstractArray; precision=Float32)
    if !isdefined(Base, :get_extension) || isnothing(Base.get_extension(@__MODULE__, :CUDAExt))
        error("GPU functionality requires CUDA.jl. Please run: using CUDA")
    end
    return _gpu_impl(x, precision)
end

"""
    cpu(x)

Transfer data to CPU. Works with states, operators, and arrays.
"""
function cpu(state::GaussianState)
    cpu_mean = isa(state.mean, AbstractArray) ? Array(state.mean) : state.mean
    cpu_covar = isa(state.covar, AbstractArray) ? Array(state.covar) : state.covar
    return GaussianState(state.basis, cpu_mean, cpu_covar; ħ = state.ħ)
end

function cpu(op::GaussianUnitary)
    cpu_disp = isa(op.disp, AbstractArray) ? Array(op.disp) : op.disp
    cpu_symplectic = isa(op.symplectic, AbstractArray) ? Array(op.symplectic) : op.symplectic
    return GaussianUnitary(op.basis, cpu_disp, cpu_symplectic; ħ = op.ħ)
end

function cpu(op::GaussianChannel)
    cpu_disp = isa(op.disp, AbstractArray) ? Array(op.disp) : op.disp
    cpu_transform = isa(op.transform, AbstractArray) ? Array(op.transform) : op.transform
    cpu_noise = isa(op.noise, AbstractArray) ? Array(op.noise) : op.noise
    return GaussianChannel(op.basis, cpu_disp, cpu_transform, cpu_noise; ħ = op.ħ)
end

function cpu(lc::GaussianLinearCombination)
    cpu_coeffs = isa(lc.coeffs, AbstractArray) ? Array(lc.coeffs) : lc.coeffs
    cpu_states = [cpu(state) for state in lc.states]
    return GaussianLinearCombination(lc.basis, cpu_coeffs, cpu_states)
end

function cpu(x::AbstractArray)
    return Array(x)
end

"""
    device(x)

Detect device of arrays/objects.
"""
device(x::AbstractArray) = _detect_device(x)
device(state::GaussianState) = _detect_device(state.mean)
device(op::GaussianUnitary) = _detect_device(op.disp)
device(op::GaussianChannel) = _detect_device(op.disp)
device(lc::GaussianLinearCombination) = _detect_device(lc)
function _detect_device(x)
    return :cpu 
end

"""
    adapt_device(target_constructor, source_obj, args...)

Create target object on same device as source.
"""
function adapt_device(target_constructor, source_obj, args...)
    source_device = device(source_obj)
    if source_device == :gpu
        if !isdefined(Base, :get_extension) || isnothing(Base.get_extension(@__MODULE__, :CUDAExt))
            @warn "GPU source object detected but CUDA extension not available. Creating CPU object instead."
            return target_constructor(args...)
        end
        return _adapt_device_gpu(target_constructor, source_obj, args...)
    else
        return target_constructor(args...)
    end
end

"""
    randstate_gpu(basis; pure=false, ħ=2, precision=Float32)

Generate random Gaussian state optimized for GPU. 
"""
function randstate_gpu(basis::SymplecticBasis; pure=false, ħ=2, precision=Float32)
    if !isdefined(Base, :get_extension) || isnothing(Base.get_extension(@__MODULE__, :CUDAExt))
        error("GPU functionality requires CUDA.jl. Please run: using CUDA")
    end
    return _randstate_gpu_impl(basis, precision; pure=pure, ħ=ħ)
end

"""
    randunitary_gpu(basis; passive=false, ħ=2, precision=Float32)

Generate random Gaussian unitary optimized for GPU. Uses hybrid CPU generation + GPU transfer.
"""
function randunitary_gpu(basis::SymplecticBasis; passive=false, ħ=2, precision=Float32)
    if !isdefined(Base, :get_extension) || isnothing(Base.get_extension(@__MODULE__, :CUDAExt))
        error("GPU functionality requires CUDA.jl. Please run: using CUDA")
    end
    return _randunitary_gpu_impl(basis, precision; passive=passive, ħ=ħ)
end

"""
    randchannel_gpu(basis; ħ=2, precision=Float32)

Generate random Gaussian channel optimized for GPU. Uses hybrid CPU generation + GPU transfer.
"""
function randchannel_gpu(basis::SymplecticBasis; ħ=2, precision=Float32)
    if !isdefined(Base, :get_extension) || isnothing(Base.get_extension(@__MODULE__, :CUDAExt))
        error("GPU functionality requires CUDA.jl. Please run: using CUDA")
    end
    return _randchannel_gpu_impl(basis, precision; ħ=ħ)
end

"""
    randsymplectic_gpu(basis; passive=false, precision=Float32)

Generate random symplectic matrix optimized for GPU. Uses hybrid CPU generation + GPU transfer.
"""
function randsymplectic_gpu(basis::SymplecticBasis; passive=false, precision=Float32)
    if !isdefined(Base, :get_extension) || isnothing(Base.get_extension(@__MODULE__, :CUDAExt))
        error("GPU functionality requires CUDA.jl. Please run: using CUDA")
    end
    return _randsymplectic_gpu_impl(basis, precision; passive=passive)
end

"""
    batch_randstate_gpu(basis, n::Int; pure=false, ħ=2, precision=Float32)

Generate n random states optimized for GPU in batch. 
"""
function batch_randstate_gpu(basis::SymplecticBasis, n::Int; pure=false, ħ=2, precision=Float32)
    if !isdefined(Base, :get_extension) || isnothing(Base.get_extension(@__MODULE__, :CUDAExt))
        error("GPU functionality requires CUDA.jl. Please run: using CUDA")
    end
    return _batch_randstate_gpu_impl(basis, n, precision; pure=pure, ħ=ħ)
end

"""
    batch_randunitary_gpu(basis, n::Int; passive=false, ħ=2, precision=Float32)

Generate n random unitaries optimized for GPU in batch.
"""
function batch_randunitary_gpu(basis::SymplecticBasis, n::Int; passive=false, ħ=2, precision=Float32)
    if !isdefined(Base, :get_extension) || isnothing(Base.get_extension(@__MODULE__, :CUDAExt))
        error("GPU functionality requires CUDA.jl. Please run: using CUDA")
    end
    return _batch_randunitary_gpu_impl(basis, n, precision; passive=passive, ħ=ħ)
end

"""
    random_ensemble_gpu(basis, n_states::Int, n_unitaries::Int; precision=Float32)

Create a complete random ensemble for GPU quantum simulations. Generates:
- n_states random pure states on GPU
- n_unitaries random unitaries on GPU
- Returns tuple (states, unitaries) ready for batched GPU operations
"""
function random_ensemble_gpu(basis::SymplecticBasis, n_states::Int, n_unitaries::Int; precision=Float32)
    if !isdefined(Base, :get_extension) || isnothing(Base.get_extension(@__MODULE__, :CUDAExt))
        error("GPU functionality requires CUDA.jl. Please run: using CUDA")
    end
    states = batch_randstate_gpu(basis, n_states; pure=true, precision=precision)
    unitaries = batch_randunitary_gpu(basis, n_unitaries; passive=false, precision=precision)
    return (states, unitaries)
end

"""
    random_simulation_setup_gpu(basis, n_samples::Int; 
                                mixed_fraction=0.3, precision=Float32)

Setup complete random simulation environment on GPU:
- Pure states (70% by default)
- Mixed states (30% by default)  
- Random channels for decoherence
- Random unitaries for dynamics

Returns NamedTuple with all components ready for GPU simulation.
"""
function random_simulation_setup_gpu(basis::SymplecticBasis, n_samples::Int; 
                                     mixed_fraction=0.3, precision=Float32)
    if !isdefined(Base, :get_extension) || isnothing(Base.get_extension(@__MODULE__, :CUDAExt))
        error("GPU functionality requires CUDA.jl. Please run: using CUDA")
    end
    n_pure = round(Int, n_samples * (1 - mixed_fraction))
    n_mixed = n_samples - n_pure
    pure_states = batch_randstate_gpu(basis, n_pure; pure=true, precision=precision)
    mixed_states = n_mixed > 0 ? batch_randstate_gpu(basis, n_mixed; pure=false, precision=precision) : typeof(pure_states[1])[]
    unitaries = batch_randunitary_gpu(basis, n_samples; passive=false, precision=precision)
    n_channels = max(1, n_samples ÷ 10)
    channels_cpu = [randchannel(basis) for _ in 1:n_channels]
    channels = [gpu(ch; precision=precision) for ch in channels_cpu]
    return (
        pure_states = pure_states,
        mixed_states = mixed_states,
        unitaries = unitaries,
        channels = channels,
        basis = basis,
        precision = precision
    )
end

function _randstate_gpu_impl end
function _randunitary_gpu_impl end  
function _randchannel_gpu_impl end
function _randsymplectic_gpu_impl end
function _batch_randstate_gpu_impl end
function _batch_randunitary_gpu_impl end
function _gpu_impl end
function _adapt_device_gpu end