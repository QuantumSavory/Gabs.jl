Base.@propagate_inbounds function _promote_output_vector(::Type{T1}, ::Type{T2}, vec_out) where {T1<:CuVector, T2<:CuVector}
    return CuArray(vec_out)
end

Base.@propagate_inbounds function _promote_output_vector(::Type{T1}, ::Type{T2}, vec_out) where {T1<:CuVector, T2<:Vector}
    return CuArray(vec_out)
end

Base.@propagate_inbounds function _promote_output_vector(::Type{T1}, ::Type{T2}, vec_out) where {T1<:Vector, T2<:CuVector}
    return CuArray(vec_out)
end

Base.@propagate_inbounds function _promote_output_vector(::Type{<:CuVector}, vec_out, vec_length::Int)
    return CuArray(vec_out)
end

Base.@propagate_inbounds function _promote_output_matrix(::Type{T1}, ::Type{T2}, mat_out) where {T1<:CuMatrix, T2<:CuMatrix}
    return CuArray(mat_out)
end

Base.@propagate_inbounds function _promote_output_matrix(::Type{T1}, ::Type{T2}, mat_out) where {T1<:CuMatrix, T2<:Matrix}
    return CuArray(mat_out)
end

Base.@propagate_inbounds function _promote_output_matrix(::Type{T1}, ::Type{T2}, mat_out) where {T1<:Matrix, T2<:CuMatrix}
    return CuArray(mat_out)
end

Base.@propagate_inbounds function _promote_output_matrix(::Type{<:CuMatrix}, mat_out, out_dim::Int)
    return CuArray(mat_out)
end

Base.@propagate_inbounds function _promote_output_matrix(::Type{<:CuMatrix}, mat_out, out_dim::Tuple)
    return CuArray(mat_out)
end

"""
    ensure_gpu_array(x::AbstractArray)

Convert array to GPU array if CUDA is available, otherwise return unchanged.
"""
function ensure_gpu_array(x::AbstractArray)
    if CUDA_AVAILABLE && !(x isa CuArray)
        return CuArray(x)
    end
    return x
end

"""
    ensure_gpu_compatibility(x::AbstractArray, target_type::Type{<:CuArray})

Ensure array is GPU-compatible with target type.
"""
function ensure_gpu_compatibility(x::AbstractArray, target_type::Type{<:CuArray})
    if CUDA_AVAILABLE
        return CuArray(x)
    else
        return Array(x)
    end
end

"""
    gpu_fallback_warning()

Issue warning when falling back to CPU due to CUDA unavailability.
"""
function gpu_fallback_warning()
    if !CUDA_AVAILABLE
        @warn "CUDA not available. Falling back to CPU computation. Install CUDA.jl and ensure GPU drivers are properly configured for GPU acceleration."
    end
end

"""
    detect_array_device_type(x::AbstractArray)

Detect device type of array using type introspection.
"""
function detect_array_device_type(x::CuArray)
    return :gpu, eltype(x), size(x)
end

function detect_array_device_type(x::AbstractArray) 
    return :cpu, eltype(x), size(x)
end

"""
    ensure_compatible_device_arrays(arrays::Tuple)

Ensure all arrays are on compatible devices. Returns device info and promotes if needed.
"""
function ensure_compatible_device_arrays(arrays::Tuple)
    if isempty(arrays)
        return :cpu, nothing
    end
    devices = [detect_array_device_type(arr)[1] for arr in arrays]
    first_device = devices[1]
    all_same = all(d -> d == first_device, devices)
    if all_same
        return first_device, arrays
    else
        has_gpu = any(d -> d == :gpu, devices)
        if has_gpu && CUDA_AVAILABLE
            promoted_arrays = ntuple(i -> arrays[i] isa CuArray ? arrays[i] : CuArray(arrays[i]), length(arrays))
            return :gpu, promoted_arrays
        else
            cpu_arrays = ntuple(i -> arrays[i] isa CuArray ? Array(arrays[i]) : arrays[i], length(arrays))
            return :cpu, cpu_arrays
        end
    end
end

"""
    validate_gpu_operation_feasibility(arrays...)

Check if GPU operation is feasible without accessing array elements.
"""
function validate_gpu_operation_feasibility(arrays...)
    if !CUDA_AVAILABLE
        return false, "CUDA not available"
    end
    total_bytes = sum(sizeof, arrays)
    if total_bytes < 1024 * 1024  
        return false, "Data size too small for GPU acceleration benefit"
    end
    return true, "GPU operation feasible"
end

"""
    get_optimal_gpu_precision(arrays...)

Determine optimal precision for GPU computation based on input arrays.
"""
function get_optimal_gpu_precision(arrays...)
    if isempty(arrays)
        return Float32  
    end
    element_types = [eltype(arr) for arr in arrays]
    if any(T -> T == Float64 || T == ComplexF64, element_types)
        return Float64
    else
        return Float32
    end
end

"""
    smart_device_transfer(obj, target_device::Symbol, precision=nothing)

Intelligently transfer object to target device with optional precision conversion.
"""
function smart_device_transfer(state::GaussianState, target_device::Symbol, precision=nothing)
    current_device = detect_array_device_type(state.mean)[1]
    if current_device == target_device
        return state  
    end
    if target_device == :gpu && CUDA_AVAILABLE
        T = precision === nothing ? eltype(state.mean) : precision
        return Gabs._gpu_impl(state, T)
    elseif target_device == :cpu
        return Gabs.cpu(state)
    else
        @warn "Invalid target device: $target_device"
        return state
    end
end

function smart_device_transfer(op::GaussianUnitary, target_device::Symbol, precision=nothing)
    current_device = detect_array_device_type(op.disp)[1]
    if current_device == target_device
        return op
    end
    if target_device == :gpu && CUDA_AVAILABLE
        T = precision === nothing ? eltype(op.disp) : precision
        return Gabs._gpu_impl(op, T)
    elseif target_device == :cpu
        return Gabs.cpu(op)
    else
        @warn "Invalid target device: $target_device"
        return op
    end
end

function smart_device_transfer(op::GaussianChannel, target_device::Symbol, precision=nothing)
    current_device = detect_array_device_type(op.disp)[1]
    if current_device == target_device
        return op
    end
    if target_device == :gpu && CUDA_AVAILABLE
        T = precision === nothing ? eltype(op.disp) : precision
        return Gabs._gpu_impl(op, T)
    elseif target_device == :cpu
        return Gabs.cpu(op)
    else
        @warn "Invalid target device: $target_device"
        return op
    end
end

"""
    batch_device_consistency_check(objects...)

Check device consistency across multiple Gabs objects.
"""
function batch_device_consistency_check(objects...)
    if isempty(objects)
        return true, :cpu, "No objects to check"
    end
    devices = []
    for obj in objects
        if obj isa GaussianState
            push!(devices, detect_array_device_type(obj.mean)[1])
        elseif obj isa GaussianUnitary
            push!(devices, detect_array_device_type(obj.disp)[1])
        elseif obj isa GaussianChannel
            push!(devices, detect_array_device_type(obj.disp)[1])
        elseif obj isa GaussianLinearCombination && !isempty(obj.states)
            push!(devices, detect_array_device_type(obj.states[1].mean)[1])
        else
            push!(devices, :cpu) 
        end
    end
    first_device = devices[1]
    all_consistent = all(d -> d == first_device, devices)
    
    if all_consistent
        return true, first_device, "All objects on same device"
    else
        return false, :mixed, "Mixed device objects detected"
    end
end

function device(x::CuArray)
    return :gpu
end