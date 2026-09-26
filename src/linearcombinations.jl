"""
    GaussianLinearCombination{B<:SymplecticBasis,C,S}

Represents a linear combination of Gaussian states of the form `Σᵢ cᵢ|ψᵢ⟩` where `cᵢ` are coefficients 
and `|ψᵢ⟩` are Gaussian states, all sharing the same symplectic basis and `ħ` value.

## Fields
- `basis::B`: Symplectic basis shared by all states
- `coeffs::Vector{C}`: Complex coefficients for the linear combination  
- `states::Vector{S}`: Vector of Gaussian states
- `ħ::Number`: Reduced Planck's constant (must be same for all states)

## Examples

julia> basis = QuadPairBasis(1)
QuadPairBasis(1)

julia> state1 = coherentstate(basis, 1.0)
GaussianState for 1 mode.
  symplectic basis: QuadPairBasis
mean: 2-element Vector{Float64}:
 2.0
 0.0
covariance: 2×2 Matrix{Float64}:
 1.0  0.0
 0.0  1.0

julia> state2 = coherentstate(basis, -1.0)
GaussianState for 1 mode.
  symplectic basis: QuadPairBasis
mean: 2-element Vector{Float64}:
 -2.0
  0.0
covariance: 2×2 Matrix{Float64}:
 1.0  0.0
 0.0  1.0

julia> cat_state = 0.5 * state1 + 0.5 * state2
GaussianLinearCombination with 2 terms for 1 mode.
  symplectic basis: QuadPairBasis
  `ħ = 2`
  [1] 0.5 * GaussianState
  [2] 0.5 * GaussianState

julia> lcgs = GaussianLinearCombination([0.6, 0.8], [state1, state2])
GaussianLinearCombination with 2 terms for 1 mode.
  symplectic basis: QuadPairBasis
  `ħ = 2`
  [1] 0.6 * GaussianState
  [2] 0.8 * GaussianState
"""
mutable struct GaussianLinearCombination{B<:SymplecticBasis,C,S}
    basis::B
    coeffs::Vector{C}
    states::Vector{S}
    ħ::Number
    function GaussianLinearCombination(basis::B, coeffs::Vector{C}, states::Vector{S}) where {B<:SymplecticBasis,C,S}
        length(coeffs) == length(states) || throw(DimensionMismatch("Number of coefficients ($(length(coeffs))) must match number of states ($(length(states)))"))
        isempty(states) && throw(ArgumentError("Cannot create an empty linear combination"))
        ħ = first(states).ħ
        @inbounds for (i, state) in enumerate(states)
            state isa GaussianState || throw(ArgumentError("Element $i is not a GaussianState: got $(typeof(state))"))
            if state.basis != basis
                throw(ArgumentError("State $i has incompatible basis: expected $(typeof(basis))($(basis.nmodes)), got $(typeof(state.basis))($(state.basis.nmodes))"))
            end
            if state.ħ != ħ
                throw(ArgumentError("State $i has different ħ: expected $ħ, got $(state.ħ)"))
            end
        end
        return new{B,C,S}(basis, coeffs, states, ħ)
    end
end
"""
    GaussianLinearCombination(state::GaussianState)

Create a linear combination containing a single Gaussian state with coefficient 1.0.
"""
function GaussianLinearCombination(state::GaussianState{B,M,V}) where {B,M,V}
    coeff_type = float(real(promote_type(eltype(M), eltype(V))))
    return GaussianLinearCombination(state.basis, [one(coeff_type)], [state])
end
"""
    GaussianLinearCombination(pairs::Vector{<:Tuple})

Create a linear combination from a vector of (coefficient, state) tuples.
"""
function GaussianLinearCombination(pairs::Vector{<:Tuple})
    isempty(pairs) && throw(ArgumentError("Cannot create an empty linear combination"))
    coeffs = [convert(Number, p[1]) for p in pairs]
    states = [p[2] for p in pairs]
    @inbounds for (i, state) in enumerate(states)
        state isa GaussianState || throw(ArgumentError("Element $i: second element must be a GaussianState"))
    end
    basis = first(states).basis
    return GaussianLinearCombination(basis, coeffs, states)
end
"""
    GaussianLinearCombination(coeffs::Vector{<:Number}, states::Vector{<:GaussianState})

Create a linear combination from separate vectors of coefficients and states.
"""
function GaussianLinearCombination(coeffs::Vector{<:Number}, states::Vector{<:GaussianState})
    isempty(states) && throw(ArgumentError("Cannot create an empty linear combination"))
    basis = first(states).basis
    return GaussianLinearCombination(basis, coeffs, states)
end
"""
    GaussianLinearCombination(pairs::Pair{<:Number,<:GaussianState}...)

Create a linear combination from coefficient => state pairs.
"""
function GaussianLinearCombination(pairs::Pair{<:Number,<:GaussianState}...)
    isempty(pairs) && throw(ArgumentError("Cannot create an empty linear combination"))
    coeffs = [p.first for p in pairs]
    states = [p.second for p in pairs]
    basis = first(states).basis
    return GaussianLinearCombination(basis, coeffs, states)
end

nmodes(x::GaussianLinearCombination) = nmodes(x[1])

"""
    +(lc1::GaussianLinearCombination, lc2::GaussianLinearCombination)

Add two linear combinations of Gaussian states. Both must have the same symplectic basis and ħ value.
"""
function Base.:+(lc1::GaussianLinearCombination{B}, lc2::GaussianLinearCombination{B}) where {B<:SymplecticBasis}
    lc1.basis == lc2.basis || throw(ArgumentError(SYMPLECTIC_ERROR))
    lc1.ħ == lc2.ħ || throw(ArgumentError(HBAR_ERROR))
    coeffs = vcat(lc1.coeffs, lc2.coeffs)
    states = vcat(lc1.states, lc2.states)
    return GaussianLinearCombination(lc1.basis, coeffs, states)
end
function Base.:+(lc1::GaussianLinearCombination{B1}, lc2::GaussianLinearCombination{B2}) where {B1<:SymplecticBasis,B2<:SymplecticBasis}
    throw(ArgumentError(SYMPLECTIC_ERROR))
end
"""
    -(lc1::GaussianLinearCombination, lc2::GaussianLinearCombination)

Subtract two linear combinations of Gaussian states.
"""
function Base.:-(lc1::GaussianLinearCombination{B}, lc2::GaussianLinearCombination{B}) where {B<:SymplecticBasis}
    return lc1 + (-1) * lc2
end
function Base.:-(lc1::GaussianLinearCombination{B1}, lc2::GaussianLinearCombination{B2}) where {B1<:SymplecticBasis,B2<:SymplecticBasis}
    throw(ArgumentError(SYMPLECTIC_ERROR))
end
"""
    -(lc::GaussianLinearCombination)

Negate a linear combination of Gaussian states.
"""
Base.:-(lc::GaussianLinearCombination) = (-1) * lc

"""
    *(α::Number, lc::GaussianLinearCombination)

Multiply a linear combination by a scalar from the left.
"""
function Base.:*(α::Number, lc::GaussianLinearCombination)
    new_coeffs = α .* lc.coeffs
    return GaussianLinearCombination(lc.basis, new_coeffs, copy(lc.states))
end
"""
    *(lc::GaussianLinearCombination, α::Number)

Multiply a linear combination by a scalar from the right.
"""
Base.:*(lc::GaussianLinearCombination, α::Number) = α * lc
"""
    *(α::Number, state::GaussianState)

Multiply a Gaussian state by a scalar to create a linear combination.
"""
function Base.:*(α::Number, state::GaussianState)
    coeff_type = promote_type(typeof(α), eltype(state.mean), eltype(state.covar))
    return GaussianLinearCombination(state.basis, [convert(coeff_type, α)], [state])
end
"""
    *(state::GaussianState, α::Number)

Multiply a Gaussian state by a scalar to create a linear combination.
"""
Base.:*(state::GaussianState, α::Number) = α * state
"""
    +(state1::GaussianState, state2::GaussianState)

Add two Gaussian states to create a linear combination.
"""
function Base.:+(state1::GaussianState, state2::GaussianState)
    state1.basis == state2.basis || throw(ArgumentError(SYMPLECTIC_ERROR))
    state1.ħ == state2.ħ || throw(ArgumentError(HBAR_ERROR))
    coeff_type = promote_type(eltype(state1.mean), eltype(state1.covar), 
                              eltype(state2.mean), eltype(state2.covar))
    return GaussianLinearCombination(state1.basis, [one(coeff_type), one(coeff_type)], [state1, state2])
end
"""
    +(state::GaussianState, lc::GaussianLinearCombination)

Add a Gaussian state to a linear combination.
"""
function Base.:+(state::GaussianState, lc::GaussianLinearCombination)
    state.basis == lc.basis || throw(ArgumentError(SYMPLECTIC_ERROR))
    state.ħ == lc.ħ || throw(ArgumentError(HBAR_ERROR))
    coeff_type = promote_type(eltype(state.mean), eltype(state.covar), eltype(lc.coeffs))
    new_coeffs = vcat(one(coeff_type), convert(Vector{coeff_type}, lc.coeffs))
    new_states = vcat(state, lc.states)
    return GaussianLinearCombination(lc.basis, new_coeffs, new_states)
end
"""
    +(lc::GaussianLinearCombination, state::GaussianState)

Add a linear combination to a Gaussian state.
"""
function Base.:+(lc::GaussianLinearCombination, state::GaussianState)
    lc.basis == state.basis || throw(ArgumentError(SYMPLECTIC_ERROR))
    lc.ħ == state.ħ || throw(ArgumentError(HBAR_ERROR))
    coeff_type = promote_type(eltype(state.mean), eltype(state.covar), eltype(lc.coeffs))
    new_coeffs = vcat(convert(Vector{coeff_type}, lc.coeffs), one(coeff_type))
    new_states = vcat(lc.states, state)
    return GaussianLinearCombination(lc.basis, new_coeffs, new_states)
end
"""
    -(state1::GaussianState, state2::GaussianState)

Subtract two Gaussian states to create a linear combination.
"""
function Base.:-(state1::GaussianState, state2::GaussianState)
    state1.basis == state2.basis || throw(ArgumentError(SYMPLECTIC_ERROR))
    state1.ħ == state2.ħ || throw(ArgumentError(HBAR_ERROR))
    coeff_type = promote_type(eltype(state1.mean), eltype(state1.covar), 
                              eltype(state2.mean), eltype(state2.covar))
    return GaussianLinearCombination(state1.basis, [one(coeff_type), -one(coeff_type)], [state1, state2])
end
"""
    -(state::GaussianState, lc::GaussianLinearCombination)

Subtract a linear combination from a Gaussian state.
"""
function Base.:-(state::GaussianState, lc::GaussianLinearCombination)
    state.basis == lc.basis || throw(ArgumentError(SYMPLECTIC_ERROR))
    state.ħ == lc.ħ || throw(ArgumentError(HBAR_ERROR))
    return state + (-1) * lc
end
"""
    -(lc::GaussianLinearCombination, state::GaussianState)

Subtract a Gaussian state from a linear combination.
"""
function Base.:-(lc::GaussianLinearCombination, state::GaussianState)
    lc.basis == state.basis || throw(ArgumentError(SYMPLECTIC_ERROR))
    lc.ħ == state.ħ || throw(ArgumentError(HBAR_ERROR))
    coeff_type = promote_type(eltype(state.mean), eltype(state.covar), eltype(lc.coeffs))
    new_coeffs = vcat(convert(Vector{coeff_type}, lc.coeffs), -one(coeff_type))
    new_states = vcat(lc.states, state)
    return GaussianLinearCombination(lc.basis, new_coeffs, new_states)
end
"""
    -(state::GaussianState)

Negate a Gaussian state to create a linear combination with coefficient -1.
"""
Base.:-(state::GaussianState) = (-1) * state

"""
    normalize!(lc::GaussianLinearCombination)

Normalize the coefficients of a linear combination in-place using L2 norm.
"""
function normalize!(lc::GaussianLinearCombination)
    norm_val = sqrt(sum(abs2, lc.coeffs))
    if norm_val > 0
        lc.coeffs ./= norm_val
    end
    return lc
end

"""
    length(lc::GaussianLinearCombination)

Return the number of terms in the linear combination.
"""
Base.length(lc::GaussianLinearCombination) = length(lc.coeffs)

"""
    getindex(lc::GaussianLinearCombination, i::Integer)

Access the i-th term as a (coefficient, state) tuple.
"""
Base.getindex(lc::GaussianLinearCombination, i::Integer) = (lc.coeffs[i], lc.states[i])

"""
    iterate(lc::GaussianLinearCombination, state=1)

Iterate over (coefficient, state) pairs in the linear combination.
"""
function Base.iterate(lc::GaussianLinearCombination, state::Int=1)
    if state > length(lc)
        return nothing
    else
        return ((lc.coeffs[state], lc.states[state]), state + 1)
    end
end

"""
    simplify!(lc::GaussianLinearCombination; atol::Real=1e-14)
    
Simplify a linear combination by:
1. Removing coefficients smaller than `atol`
2. Combining terms with identical states
3. Ensuring at least one term remains (with minimal coefficient vacuum state if all coefficients become negligible)
Returns the modified linear combination.
"""
function simplify!(lc::GaussianLinearCombination; atol::Real=1e-14)
    if isempty(lc.coeffs)
        return lc
    end
    keep_mask = abs.(lc.coeffs) .> atol
    if !any(keep_mask)
        vac = vacuumstate(lc.basis, ħ = lc.ħ)
        coeff_type = eltype(lc.coeffs)
        lc.coeffs = [coeff_type(atol)] 
        lc.states = [vac]
        return lc
    end
    coeffs = lc.coeffs[keep_mask]
    states = lc.states[keep_mask]
    unique_states = Vector{typeof(states[1])}(undef, length(states))
    combined_coeffs = Vector{eltype(coeffs)}(undef, length(states))
    n_unique = 0  
    for (coeff, state) in zip(coeffs, states)
        existing_idx = findfirst(s -> isapprox(s, state, atol=1e-12), @view(unique_states[1:n_unique]))
        if existing_idx === nothing
            n_unique += 1
            unique_states[n_unique] = state
            combined_coeffs[n_unique] = coeff
        else
            combined_coeffs[existing_idx] += coeff
        end
    end
    resize!(unique_states, n_unique)
    resize!(combined_coeffs, n_unique)
    final_mask = abs.(combined_coeffs) .> atol
    if !any(final_mask)
        vac = vacuumstate(lc.basis, ħ = lc.ħ)
        coeff_type = eltype(combined_coeffs)
        lc.coeffs = [coeff_type(atol)]
        lc.states = [vac]
    else
        lc.coeffs = combined_coeffs[final_mask]
        lc.states = unique_states[final_mask]
    end
    return lc
end

function Base.show(io::IO, mime::MIME"text/plain", lc::GaussianLinearCombination)
    basis_name = nameof(typeof(lc.basis))
    nmodes = lc.basis.nmodes
    print(io, "GaussianLinearCombination with $(length(lc)) terms")
    if nmodes == 1
        println(io, " for 1 mode.")
    else
        println(io, " for $(nmodes) modes.")
    end
    println(io, "  symplectic basis: ", basis_name)
    println(io, "  ħ = $(lc.ħ)")
    max_display = min(length(lc), 5)
    @inbounds for i in 1:max_display
        coeff, state = lc[i]
        println(io, "  [$i] $(coeff) * GaussianState")
    end
    if length(lc) > max_display
        println(io, "  ⋮ ($(length(lc) - max_display) more terms)")
    end
end

Base.show(io::IO, lc::GaussianLinearCombination) = print(io, "GaussianLinearCombination($(length(lc)) terms)")

function Base.:(==)(lc1::GaussianLinearCombination, lc2::GaussianLinearCombination)
    lc1.basis == lc2.basis || return false
    lc1.ħ == lc2.ħ || return false
    length(lc1) == length(lc2) || return false
    return lc1.coeffs == lc2.coeffs && all(isequal(s1, s2) for (s1, s2) in zip(lc1.states, lc2.states))
end

function Base.isapprox(lc1::GaussianLinearCombination, lc2::GaussianLinearCombination; kwargs...)
    lc1.basis == lc2.basis || return false
    lc1.ħ == lc2.ħ || return false
    length(lc1) == length(lc2) || return false
    return isapprox(lc1.coeffs, lc2.coeffs; kwargs...) && 
           all(isapprox(s1, s2; kwargs...) for (s1, s2) in zip(lc1.states, lc2.states))
end

"""
    *(op::GaussianUnitary, lc::GaussianLinearCombination)

Apply a Gaussian unitary operation to a linear combination of Gaussian states.
The unitary is applied to each component state while preserving coefficients.
"""
function Base.:(*)(op::GaussianUnitary, lc::GaussianLinearCombination)
    op.basis == lc.basis || throw(ArgumentError(ACTION_ERROR))
    op.ħ == lc.ħ || throw(ArgumentError(HBAR_ERROR))
    new_states = [op * state for state in lc.states]
    return GaussianLinearCombination(lc.basis, copy(lc.coeffs), new_states)
end

"""
    *(op::GaussianChannel, lc::GaussianLinearCombination)

Apply a Gaussian channel to a linear combination of Gaussian states.
The channel is applied to each component state while preserving coefficients.
"""
function Base.:(*)(op::GaussianChannel, lc::GaussianLinearCombination)
    op.basis == lc.basis || throw(ArgumentError(ACTION_ERROR))
    op.ħ == lc.ħ || throw(ArgumentError(HBAR_ERROR))
    new_states = [op * state for state in lc.states]
    return GaussianLinearCombination(lc.basis, copy(lc.coeffs), new_states)
end

"""
    tensor(::Type{Tc}, ::Type{Ts}, lc1::GaussianLinearCombination, lc2::GaussianLinearCombination)

Compute tensor product of two linear combinations with specified output types.
Creates all pairwise tensor products: `Σᵢⱼ cᵢcⱼ |ψᵢ⟩ ⊗ |ϕⱼ⟩`.
"""
function tensor(::Type{Tm}, ::Type{Tc}, lc1::GaussianLinearCombination, lc2::GaussianLinearCombination) where {Tm,Tc}
    typeof(lc1.basis) == typeof(lc2.basis) || throw(ArgumentError(SYMPLECTIC_ERROR))
    lc1.ħ == lc2.ħ || throw(ArgumentError(HBAR_ERROR))
    new_basis = lc1.basis ⊕ lc2.basis
    n1, n2 = length(lc1), length(lc2)
    CoeffType = promote_type(eltype(lc1.coeffs), eltype(lc2.coeffs))
    new_coeffs = Vector{CoeffType}(undef, n1 * n2)
    new_states = Vector{GaussianState}(undef, n1 * n2)
    @inbounds for i in 1:n1
        @inbounds for j in 1:n2
            idx = (i-1) * n2 + j
            new_coeffs[idx] = lc1.coeffs[i] * lc2.coeffs[j]
            new_states[idx] = tensor(Tm, Tc, lc1.states[i], lc2.states[j])
        end
    end
    return GaussianLinearCombination(new_basis, new_coeffs, new_states)
end
function tensor(::Type{T}, lc1::GaussianLinearCombination, lc2::GaussianLinearCombination) where {T}
    if T <: AbstractMatrix
        return tensor(Vector{eltype(T)}, T, lc1, lc2)
    else
        return tensor(T, T, lc1, lc2)
    end
end
"""
    tensor(lc1::GaussianLinearCombination, lc2::GaussianLinearCombination)

Compute tensor product of two linear combinations of Gaussian states.
"""
function tensor(lc1::GaussianLinearCombination, lc2::GaussianLinearCombination)
    typeof(lc1.basis) == typeof(lc2.basis) || throw(ArgumentError(SYMPLECTIC_ERROR))
    lc1.ħ == lc2.ħ || throw(ArgumentError(HBAR_ERROR))
    new_basis = lc1.basis ⊕ lc2.basis
    n1, n2 = length(lc1), length(lc2)
    CoeffType = promote_type(eltype(lc1.coeffs), eltype(lc2.coeffs))
    new_coeffs = Vector{CoeffType}(undef, n1 * n2)
    new_states = Vector{GaussianState}(undef, n1 * n2)
    @inbounds for i in 1:n1
        @inbounds for j in 1:n2
            idx = (i-1) * n2 + j
            new_coeffs[idx] = lc1.coeffs[i] * lc2.coeffs[j]
            new_states[idx] = lc1.states[i] ⊗ lc2.states[j]
        end
    end
    return GaussianLinearCombination(new_basis, new_coeffs, new_states)
end
function _tensor(lc1::GaussianLinearCombination, lc2::GaussianLinearCombination)
    @assert typeof(lc1.basis) == typeof(lc2.basis) "Linear combinations must have compatible bases"
    @assert lc1.ħ == lc2.ħ "Linear combinations must have the same ħ"
    new_basis = lc1.basis ⊕ lc2.basis
    result_size = length(lc1) * length(lc2)
    CoeffType = promote_type(eltype(lc1.coeffs), eltype(lc2.coeffs))
    new_coeffs = Vector{CoeffType}(undef, result_size)
    new_states = Vector{GaussianState}(undef, result_size)
    idx = 1
    @inbounds for (c1, s1) in lc1
        @inbounds for (c2, s2) in lc2
            new_coeffs[idx] = c1 * c2
            new_states[idx] = s1 ⊗ s2
            idx += 1
        end
    end
    return GaussianLinearCombination(new_basis, new_coeffs, new_states)
end

function ptrace(::Type{Tm}, ::Type{Tc}, lc::GaussianLinearCombination, index::Int) where {Tm,Tc}
    length([index]) < lc.basis.nmodes || throw(ArgumentError(INDEX_ERROR))
    traced_states = [ptrace(Tm, Tc, state, [index]) for state in lc.states]
    result = GaussianLinearCombination(traced_states[1].basis, copy(lc.coeffs), traced_states)
    simplify!(result)
    return result
end
function ptrace(::Type{Tm}, ::Type{Tc}, lc::GaussianLinearCombination, indices::AbstractVector{<:Int}) where {Tm,Tc}
    length(indices) < lc.basis.nmodes || throw(ArgumentError(INDEX_ERROR))
    traced_states = [ptrace(Tm, Tc, state, indices) for state in lc.states]
    result = GaussianLinearCombination(traced_states[1].basis, copy(lc.coeffs), traced_states)
    simplify!(result)
    return result
end
function ptrace(::Type{T}, lc::GaussianLinearCombination, indices::Union{Int, AbstractVector{<:Int}}) where {T}
    if T <: AbstractMatrix
        return ptrace(Vector{eltype(T)}, T, lc, indices)
    else
        return ptrace(T, T, lc, indices)
    end
end
"""
    ptrace(lc::GaussianLinearCombination, index::Int)

Compute partial trace of a linear combination over specified index.
"""
function ptrace(lc::GaussianLinearCombination, index::Int)
    length([index]) < lc.basis.nmodes || throw(ArgumentError(INDEX_ERROR))
    traced_states = [ptrace(state, [index]) for state in lc.states]
    result = GaussianLinearCombination(traced_states[1].basis, copy(lc.coeffs), traced_states)
    simplify!(result)
    return result
end
"""
    ptrace(lc::GaussianLinearCombination, indices::AbstractVector{<:Int})

Compute partial trace of a linear combination over specified indices.
Combines identical traced states automatically.
"""
function ptrace(lc::GaussianLinearCombination, indices::AbstractVector{<:Int})
    length(indices) < lc.basis.nmodes || throw(ArgumentError(INDEX_ERROR))
    traced_states = [ptrace(state, indices) for state in lc.states]
    result = GaussianLinearCombination(traced_states[1].basis, copy(lc.coeffs), traced_states)
    simplify!(result)
    return result
end

"""
    embed(basis::SymplecticBasis, idx::Int, lc::GaussianLinearCombination)
    embed(basis::SymplecticBasis, indices::Vector{<:Int}, lc::GaussianLinearCombination)

Embed every state in a `GaussianLinearCombination` into a larger symplectic
basis while preserving coefficients.
"""
function embed(basis::SymplecticBasis, index::Int, lc::GaussianLinearCombination)
    return embed(basis, [index], lc)
end
function embed(basis::SymplecticBasis, indices::Vector{<:Int}, lc::GaussianLinearCombination)
    @assert length(indices) == lc.basis.nmodes "Number of indices must match number of modes in the linear combination"
    @assert basis.nmodes ≥ length(indices) "Target basis must be large enough"

    embedded_states = [embed(basis, indices, state) for state in lc.states]
    return GaussianLinearCombination(basis, copy(lc.coeffs), embedded_states)
end

"""
    cross_wigner(state1::GaussianState, state2::GaussianState, x::AbstractVector)

Compute the off-diagonal Wigner kernel (cross-Wigner function) between two Gaussian states.

The cross-Wigner function is given by:

    `W₁₂(x) = (1/(2π)ⁿ√det((V₁+V₂)/2)) × exp[-½(x-μ̄)ᵀ((V₁+V₂)/2)⁻¹(x-μ̄)] × exp[i(μ₁-μ₂)ᵀΩ(x-μ̄)/ħ]`

where:
- `μ̄ = (μ₁ + μ₂)/2` is the average of the two mean vectors
- `μ₁`, `μ₂` are the mean vectors of the two states  
- `V₁`, `V₂` are the covariance matrices of the two states
- `Ω` is the symplectic form matrix
- `n` is the number of modes
- `ħ` is the reduced Planck constant

This function captures the quantum interference between two Gaussian states and is 
essential for computing Wigner functions of superposition states. The cross-Wigner 
function appears in the interference terms when computing the Wigner function of 
a linear combination `|ψ⟩ = c₁|ψ₁⟩ + c₂|ψ₂⟩`:

    `W(x) = |c₁|²W₁(x) + |c₂|²W₂(x) + 2Re(c₁*c₂*W₁₂(x))`

The normalization is chosen to ensure the identity property `W₁₁(x) = W₁(x)`.

# Arguments
- `state1::GaussianState`: First Gaussian state
- `state2::GaussianState`: Second Gaussian state  
- `x::AbstractVector`: Phase space point where to evaluate the function

# Returns
- `ComplexF64`: Complex value of the cross-Wigner function at point `x`

# Notes
- The function is Hermitian: `W₁₂(x) = W₂₁*(x)`
- For identical states: `W₁₁(x) = W₁(x)` (reduces to regular Wigner function)
- The implementation uses log-space arithmetic for numerical stability
"""
function cross_wigner(state1::GaussianState, state2::GaussianState, x::AbstractVector)
    μ1, μ2 = state1.mean, state2.mean
    n = length(μ1) ÷ 2
    ħ = state1.ħ
    Vavg = (state1.covar .+ state2.covar) ./ 2
    dx = x .- (μ1 .+ μ2) ./ 2
    Ω = _symplecticform(state1.basis, Vavg)
    phase_arg = dot(μ1 .- μ2, Ω * dx)
    lognorm = -n*log(2π) - 0.5*_logdet(Vavg)
    gauss = exp(-0.5 * dot(dx, Vavg \ dx))
    return exp(lognorm) * gauss * cis(phase_arg / ħ)
end

"""
    cross_wigner(state1::GaussianState, state2::GaussianState, xs::AbstractMatrix)

Compute the cross-Wigner function at each column of `xs`, a matrix of size
2N x M.
"""
function cross_wigner(state1::GaussianState, state2::GaussianState, xs::AbstractMatrix)
    state1.basis == state2.basis || throw(ArgumentError(SYMPLECTIC_ERROR))
    state1.ħ == state2.ħ || throw(ArgumentError(HBAR_ERROR))
    size(xs, 1) == length(state1.mean) || throw(ArgumentError(WIGNER_ERROR))
    μ1, μ2 = state1.mean, state2.mean
    n = length(μ1) ÷ 2
    ħ = state1.ħ
    Vavg = (state1.covar .+ state2.covar) ./ 2
    Ω = _symplecticform(state1.basis, Vavg)
    dx = xs .- (μ1 .+ μ2) ./ 2
    quad = vec(sum(dx .* (Vavg \ dx); dims = 1))
    phase_arg = vec(transpose(Ω * (μ1 .- μ2)) * dx)
    lognorm = -n*log(2π) - 0.5*_logdet(Vavg)
    # Ω is antisymmetric, hence the sign on the phase
    return exp(lognorm) .* exp.(-0.5 .* quad) .* cis.(.-phase_arg ./ ħ)
end

"""
    wigner(lc::GaussianLinearCombination, x::AbstractVector)

Compute Wigner function of a linear combination including quantum interference.
`W(x) = Σᵢ |cᵢ|² Wᵢ(x) + 2 Σᵢ<ⱼ Re(cᵢ*cⱼ W_cross(ψᵢ,ψⱼ,x))`
"""
function wigner(lc::GaussianLinearCombination, x::AbstractVector)
    length(x) == length(lc.states[1].mean) || throw(ArgumentError(WIGNER_ERROR))
    # seeded from the first term, so the sum takes the states' element type
    c1, s1 = lc[1]
    result = abs2(c1) * wigner(s1, x)
    @inbounds for i in 1:length(lc)
        ci, si = lc[i]
        i > 1 && (result += abs2(ci) * wigner(si, x))
        for j in (i+1):length(lc)
            cj, sj = lc[j]
            result += 2 * real(conj(ci) * cj * cross_wigner(si, sj, x))
        end
    end
    return result
end

"""
    wigner(lc::GaussianLinearCombination, xs::AbstractMatrix)

Compute the Wigner function of a linear combination, including interference, at
each column of `xs`.
"""
function wigner(lc::GaussianLinearCombination, xs::AbstractMatrix)
    size(xs, 1) == length(lc.states[1].mean) || throw(ArgumentError(WIGNER_ERROR))
    c1, s1 = lc[1]
    result = abs2(c1) .* wigner(s1, xs)
    @inbounds for i in 1:length(lc)
        ci, si = lc[i]
        i > 1 && (result .+= abs2(ci) .* wigner(si, xs))
        for j in (i+1):length(lc)
            cj, sj = lc[j]
            result .+= 2 .* real.(conj(ci) * cj .* cross_wigner(si, sj, xs))
        end
    end
    return result
end

"""
    cross_wignerchar(state1::GaussianState, state2::GaussianState, xi::AbstractVector)

Compute cross-Wigner characteristic function between two Gaussian states.
"""
function cross_wignerchar(state1::GaussianState, state2::GaussianState, xi::AbstractVector)
    state1.basis == state2.basis || throw(ArgumentError(SYMPLECTIC_ERROR))
    state1.ħ == state2.ħ || throw(ArgumentError(HBAR_ERROR))
    length(xi) == length(state1.mean) || throw(ArgumentError(WIGNER_ERROR))
    if state1 === state2
        return wignerchar(state1, xi)
    end
    μ12 = (state1.mean .+ state2.mean) ./ 2
    V12 = (state1.covar .+ state2.covar) ./ 2
    Omega = _symplecticform(state1.basis, V12)
    M = -(Omega * V12 * Omega)
    arg1 = -0.5 * dot(xi, M * xi)
    arg2 = 1im * dot(Omega * μ12, xi)
    return exp(arg1 - arg2)
end

"""
    cross_wignerchar(state1::GaussianState, state2::GaussianState, xis::AbstractMatrix)

Compute the cross-Wigner characteristic function at each column of `xis`, a
matrix of size 2N x M.
"""
function cross_wignerchar(state1::GaussianState, state2::GaussianState, xis::AbstractMatrix)
    state1.basis == state2.basis || throw(ArgumentError(SYMPLECTIC_ERROR))
    state1.ħ == state2.ħ || throw(ArgumentError(HBAR_ERROR))
    size(xis, 1) == length(state1.mean) || throw(ArgumentError(WIGNER_ERROR))
    μ12 = (state1.mean .+ state2.mean) ./ 2
    V12 = (state1.covar .+ state2.covar) ./ 2
    Ω = _symplecticform(state1.basis, V12)
    M = -(Ω * V12 * Ω)
    args1 = -(1/2) .* vec(sum(xis .* (M * xis); dims = 1))
    args2 = im .* vec(transpose(Ω * μ12) * xis)
    return exp.(args1 .- args2)
end

"""
    wignerchar(lc::GaussianLinearCombination, xis::AbstractMatrix)

Compute the Wigner characteristic function of a linear combination, including
interference, at each column of `xis`.
"""
function wignerchar(lc::GaussianLinearCombination, xis::AbstractMatrix)
    size(xis, 1) == length(lc.states[1].mean) || throw(ArgumentError(WIGNER_ERROR))
    c1, s1 = lc[1]
    result = complex.(abs2(c1) .* wignerchar(s1, xis))
    @inbounds for i in 1:length(lc)
        ci, si = lc[i]
        i > 1 && (result .+= abs2(ci) .* wignerchar(si, xis))
        for j in (i+1):length(lc)
            cj, sj = lc[j]
            result .+= 2 .* real.(conj(ci) * cj .* cross_wignerchar(si, sj, xis))
        end
    end
    return result
end

"""
    wignerchar(lc::GaussianLinearCombination, xi::AbstractVector)

Compute Wigner characteristic function of a linear combination including interference.
"""
function wignerchar(lc::GaussianLinearCombination, xi::AbstractVector)
    length(xi) == length(lc.states[1].mean) || throw(ArgumentError(WIGNER_ERROR))
    c1, s1 = lc[1]
    result = complex(abs2(c1) * wignerchar(s1, xi))
    @inbounds for i in 1:length(lc)
        ci, si = lc[i]
        i > 1 && (result += abs2(ci) * wignerchar(si, xi))
        for j in (i+1):length(lc)
            cj, sj = lc[j]
            result += 2 * real(conj(ci) * cj * cross_wignerchar(si, sj, xi))
        end
    end
    return result
end

function purity(lc::GaussianLinearCombination)
    # A GaussianLinearCombination represents a pure superposition state
    # |ψ⟩ = Σᵢ cᵢ|ψᵢ⟩, so purity = Tr(ρ²) = 1
    return 1.0
end

function entropy_vn(lc::GaussianLinearCombination)
    # A GaussianLinearCombination represents a pure superposition state
    # |ψ⟩ = Σᵢ cᵢ|ψᵢ⟩, so S(ρ) = -Tr(ρ log ρ) = 0
    return 0.0
end