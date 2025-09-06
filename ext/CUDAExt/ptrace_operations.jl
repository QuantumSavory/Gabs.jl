"""
    ptrace(state::GaussianState{B,M,V}, indices::Union{Int, AbstractVector{<:Int}}) where {B<:SymplecticBasis, M<:CuArray, V<:CuArray}
"""
function ptrace(state::GaussianState{B,M,V}, indices::Union{Int, AbstractVector{<:Int}}) where {
    B<:SymplecticBasis, M<:CuArray, V<:CuArray}
    if !CUDA_AVAILABLE
        gpu_fallback_warning()
        cpu_state = GaussianState(state.basis, Array(state.mean), Array(state.covar); ħ=state.ħ)
        cpu_result = ptrace(cpu_state, indices)
        return GaussianState(cpu_result.basis, CuArray(cpu_result.mean), CuArray(cpu_result.covar); ħ=cpu_result.ħ)
    end
    indices_vec = isa(indices, Int) ? [indices] : collect(indices)
    basis = state.basis
    nmodes = basis.nmodes
    length(indices_vec) < nmodes || throw(ArgumentError(Gabs.INDEX_ERROR))
    all_indices = collect(1:nmodes)
    keep_indices = setdiff(all_indices, indices_vec)
    n_keep = length(keep_indices)
    T = eltype(M)
    if basis isa QuadPairBasis
        mean_indices = Vector{Int}()
        for mode_idx in keep_indices
            push!(mean_indices, 2*mode_idx - 1)  
            push!(mean_indices, 2*mode_idx)     
        end
        new_mean = state.mean[CuArray(mean_indices)]
        covar_indices = Vector{Int}()
        for mode_idx in keep_indices
            push!(covar_indices, 2*mode_idx - 1) 
            push!(covar_indices, 2*mode_idx)     
        end
        covar_indices_gpu = CuArray(covar_indices)
        new_covar = state.covar[covar_indices_gpu, covar_indices_gpu]
    elseif basis isa QuadBlockBasis
        x_indices = keep_indices
        p_indices = keep_indices .+ nmodes
        mean_indices = vcat(x_indices, p_indices)
        new_mean = state.mean[CuArray(mean_indices)]
        all_covar_indices = vcat(x_indices, p_indices)
        covar_indices_gpu = CuArray(all_covar_indices)
        new_covar = state.covar[covar_indices_gpu, covar_indices_gpu]
    else
        error("Unknown basis type: $(typeof(basis))")
    end
    new_basis = typeof(basis)(n_keep)
    return GaussianState(new_basis, new_mean, new_covar; ħ = state.ħ)
end

function ptrace(::Type{Tm}, ::Type{Tc}, state::GaussianState{B,<:CuArray,<:CuArray}, indices::Union{Int, AbstractVector{<:Int}}) where {Tm<:CuArray, Tc<:CuArray, B<:SymplecticBasis}
    result = ptrace(state, indices)
    return GaussianState(result.basis, Tm(result.mean), Tc(result.covar); ħ = result.ħ)
end

function ptrace(::Type{T}, state::GaussianState{B,<:CuArray,<:CuArray}, indices::Union{Int, AbstractVector{<:Int}}) where {T<:CuArray, B<:SymplecticBasis}
    if T <: AbstractMatrix
        return ptrace(CuVector{eltype(T)}, T, state, indices)
    else
        return ptrace(T, T, state, indices)
    end
end