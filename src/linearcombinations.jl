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
    coeffs::C 
    states::Vector{S}
    ħ::Number
    function GaussianLinearCombination(basis::B, coeffs::AbstractVector{C}, states::Vector{S}) where {B<:SymplecticBasis,C<:Number,S}                                                   
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
        return new{B,typeof(coeffs),S}(basis, coeffs, states, ħ)
                         
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
function GaussianLinearCombination(coeffs::AbstractVector{<:Number}, states::Vector{<:GaussianState})
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
    lc1_has_gpu_states = !isempty(lc1.states) && _is_gpu_array(lc1.states[1].mean)
    lc2_has_gpu_states = !isempty(lc2.states) && _is_gpu_array(lc2.states[1].mean)
    promote_to_gpu = lc1_has_gpu_states || lc2_has_gpu_states
    if promote_to_gpu
        T = promote_type(eltype(lc1.coeffs), eltype(lc2.coeffs))
        coeffs1 = Vector{T}(Array(lc1.coeffs))  
        coeffs2 = Vector{T}(Array(lc2.coeffs))  
        if lc1_has_gpu_states
            states1 = lc1.states
        else
            precision = T <: Complex ? real(T) : T
            states1 = [gpu(s, precision = precision) for s in lc1.states]
        end
        if lc2_has_gpu_states
            states2 = lc2.states
        else
            precision = T <: Complex ? real(T) : T
            states2 = [gpu(s, precision = precision) for s in lc2.states]
        end
        coeffs = vcat(coeffs1, coeffs2)
        states = vcat(states1, states2)
    else
        T = promote_type(eltype(lc1.coeffs), eltype(lc2.coeffs))
        coeffs1 = Vector{T}(lc1.coeffs)
        coeffs2 = Vector{T}(lc2.coeffs)
        coeffs = vcat(coeffs1, coeffs2)
        states = vcat(lc1.states, lc2.states)
    end
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
    T = promote_type(typeof(α), eltype(lc.coeffs))
    new_coeffs = Vector{T}(α .* Array(lc.coeffs)) 
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
    T = promote_type(typeof(α), eltype(state.mean), eltype(state.covar))
    coeff_array = Vector{T}([convert(T, α)])  
    return GaussianLinearCombination(state.basis, coeff_array, [state])
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
    state1_gpu = _is_gpu_array(state1.mean)
    state2_gpu = _is_gpu_array(state2.mean)
    T = promote_type(eltype(state1.mean), eltype(state1.covar), 
                     eltype(state2.mean), eltype(state2.covar))
    coeffs = Vector{T}([one(T), one(T)])
    if state1_gpu || state2_gpu
        precision = T <: Complex ? real(T) : T
        states = [
            state1_gpu ? state1 : gpu(state1, precision = precision),
            state2_gpu ? state2 : gpu(state2, precision = precision)
        ]
    else
        states = [state1, state2]
    end
    
    return GaussianLinearCombination(state1.basis, coeffs, states)
end
"""
    +(state::GaussianState, lc::GaussianLinearCombination)

Add a Gaussian state to a linear combination.
"""
function Base.:+(state::GaussianState, lc::GaussianLinearCombination)
    state.basis == lc.basis || throw(ArgumentError(SYMPLECTIC_ERROR))
    state.ħ == lc.ħ || throw(ArgumentError(HBAR_ERROR))
    state_gpu = _is_gpu_array(state.mean)
    lc_has_gpu = !isempty(lc.states) && _is_gpu_array(lc.states[1].mean)
    T = promote_type(eltype(state.mean), eltype(state.covar), eltype(lc.coeffs))
    new_coeffs = Vector{T}(vcat([one(T)], Array(lc.coeffs)))
    if state_gpu || lc_has_gpu
        precision = T <: Complex ? real(T) : T
        promoted_state = state_gpu ? state : gpu(state, precision = precision)
        promoted_states = lc_has_gpu ? lc.states : [gpu(s, precision = precision) for s in lc.states]
        new_states = vcat([promoted_state], promoted_states)
    else
        new_states = vcat([state], lc.states)
    end
    
    return GaussianLinearCombination(lc.basis, new_coeffs, new_states)
end
"""
    +(lc::GaussianLinearCombination, state::GaussianState)

Add a linear combination to a Gaussian state.
"""
function Base.:+(lc::GaussianLinearCombination, state::GaussianState)
    lc.basis == state.basis || throw(ArgumentError(SYMPLECTIC_ERROR))
    lc.ħ == state.ħ || throw(ArgumentError(HBAR_ERROR))
    state_gpu = _is_gpu_array(state.mean)
    lc_has_gpu = !isempty(lc.states) && _is_gpu_array(lc.states[1].mean)
    T = promote_type(eltype(state.mean), eltype(state.covar), eltype(lc.coeffs))
    new_coeffs = Vector{T}(vcat(Array(lc.coeffs), [one(T)]))
    if state_gpu || lc_has_gpu
        precision = T <: Complex ? real(T) : T
        promoted_state = state_gpu ? state : gpu(state, precision = precision)
        promoted_states = lc_has_gpu ? lc.states : [gpu(s, precision = precision) for s in lc.states]
        new_states = vcat(promoted_states, [promoted_state])
    else
        new_states = vcat(lc.states, [state])
    end
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
    cpu_coeffs = Vector(Array(lc.coeffs))  
    lc_has_gpu_states = !isempty(lc.states) && _is_gpu_array(lc.states[1].mean)
    keep_mask = abs.(cpu_coeffs) .> atol
    if !any(keep_mask)
        vac = _device_aware_vacuum(lc)
        coeff_type = eltype(cpu_coeffs)
        lc.coeffs = Vector{coeff_type}([coeff_type(atol)]) 
        lc.states = [vac]
        return lc
    end
    surviving_coeffs = cpu_coeffs[keep_mask]
    surviving_states = lc.states[keep_mask] 
    if isempty(surviving_states)
        return lc
    end
    unique_states = Vector{typeof(surviving_states[1])}()
    combined_coeffs = Vector{eltype(surviving_coeffs)}()
    
    for (coeff, state) in zip(surviving_coeffs, surviving_states)
        existing_idx = findfirst(unique_states) do existing_state
            return states_approximately_equal(existing_state, state, 1e-12)
        end
        if existing_idx === nothing
            push!(unique_states, state)
            push!(combined_coeffs, coeff)
        else
            combined_coeffs[existing_idx] += coeff
        end
    end
    final_mask = abs.(combined_coeffs) .> atol
    if !any(final_mask)
        vac = _device_aware_vacuum(lc)
        coeff_type = eltype(combined_coeffs)
        lc.coeffs = Vector{coeff_type}([coeff_type(atol)]) 
        lc.states = [vac]
    else
        final_coeffs = combined_coeffs[final_mask]
        lc.coeffs = Vector{eltype(final_coeffs)}(final_coeffs) 
        lc.states = unique_states[final_mask]
    end
    return lc
end

"""
    states_approximately_equal(state1::GaussianState, state2::GaussianState, atol::Real)
"""
function states_approximately_equal(state1::GaussianState, state2::GaussianState, atol::Real)
    state1.basis != state2.basis && return false
    state1.ħ != state2.ħ && return false
    try
        mean_diff = maximum(abs.(Array(state1.mean) .- Array(state2.mean)))
        covar_diff = maximum(abs.(Array(state1.covar) .- Array(state2.covar)))
        return mean_diff <= atol && covar_diff <= atol
    catch e
        return false
    end
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
    op_gpu = _is_gpu_array(op.disp)
    lc_gpu = (device(lc) == :gpu)
    if op_gpu && !lc_gpu
        T = real(eltype(op.disp))
        gpu_states = [gpu(s, precision = T) for s in lc.states]
        new_states = [op * state for state in gpu_states]
        return GaussianLinearCombination(lc.basis, copy(lc.coeffs), new_states)
    elseif !op_gpu && lc_gpu
        T = real(eltype(lc.coeffs))
        gpu_op = gpu(op, precision = T)
        new_states = [gpu_op * state for state in lc.states]
        return GaussianLinearCombination(lc.basis, copy(lc.coeffs), new_states)
    else
        new_states = [op * state for state in lc.states]
        return GaussianLinearCombination(lc.basis, copy(lc.coeffs), new_states)
    end
end

"""
    *(op::GaussianChannel, lc::GaussianLinearCombination)

Apply a Gaussian channel to a linear combination of Gaussian states.
The channel is applied to each component state while preserving coefficients.
"""
function Base.:(*)(op::GaussianChannel, lc::GaussianLinearCombination)
    op.basis == lc.basis || throw(ArgumentError(ACTION_ERROR))
    op.ħ == lc.ħ || throw(ArgumentError(HBAR_ERROR))
    op_gpu = _is_gpu_array(op.disp)
    lc_has_gpu = !isempty(lc.states) && _is_gpu_array(lc.states[1].mean)
    cpu_coeffs = Vector(Array(lc.coeffs))
    if op_gpu && !lc_has_gpu
        T = real(eltype(op.disp))
        gpu_states = [gpu(s, precision = T) for s in lc.states]
        new_states = [op * state for state in gpu_states]
    elseif !op_gpu && lc_has_gpu
        T = real(eltype(lc.states[1].mean))
        gpu_op = gpu(op, precision = T)
        new_states = [gpu_op * state for state in lc.states]
    else
        new_states = [op * state for state in lc.states]
    end
    return GaussianLinearCombination(lc.basis, cpu_coeffs, new_states)
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
    V1, V2 = state1.covar, state2.covar
    n = length(μ1) ÷ 2
    ħ = state1.ħ
    dx = similar(x)
    Vavg = similar(V1)
    @inbounds @. dx = x - 0.5 * (μ1 + μ2)
    @inbounds @. Vavg = 0.5 * (V1 + V2)
    Ω = symplecticform(state1.basis)
    phase_arg = zero(real(eltype(μ1)))
    @inbounds for i in eachindex(μ1)
        Δμi = μ1[i] - μ2[i]
        Ω_dx_i = zero(eltype(dx))
        @simd for j in eachindex(dx)
            Ω_dx_i += Ω[i,j] * dx[j]
        end
        phase_arg += Δμi * Ω_dx_i
    end
    phase = cis(phase_arg / ħ)
    lognorm = -n*log(2π) - 0.5*logdet(Vavg)
    norm = exp(lognorm)
    gauss = exp(-0.5 * dot(dx, Vavg \ dx))  
    return norm * gauss * phase
end

"""
    wigner(lc::GaussianLinearCombination, x::AbstractVector)

Compute Wigner function of a linear combination including quantum interference.
`W(x) = Σᵢ |cᵢ|² Wᵢ(x) + 2 Σᵢ<ⱼ Re(cᵢ*cⱼ W_cross(ψᵢ,ψⱼ,x))`
"""
function wigner(lc::GaussianLinearCombination, x::AbstractVector)
    length(x) == length(lc.states[1].mean) || throw(ArgumentError(WIGNER_ERROR))
    result = 0.0
    @inbounds for i in 1:length(lc)
        ci, si = lc[i]
        result += abs2(ci) * wigner(si, x)
        @inbounds for j in (i+1):length(lc)
            cj, sj = lc[j]
            cross_term = 2 * real(conj(ci) * cj * cross_wigner(si, sj, x))
            result += cross_term
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
    μ1, μ2 = state1.mean, state2.mean
    V1, V2 = state1.covar, state2.covar
    μ12 = similar(μ1)
    V12 = similar(V1)
    temp_vec = similar(μ1)
    temp_mat = similar(V1)
    @inbounds @simd for i in eachindex(μ1)
        μ12[i] = (μ1[i] + μ2[i]) * 0.5  
    end
    @inbounds @simd for i in eachindex(V1)
        V12[i] = (V1[i] + V2[i]) * 0.5
    end
    Omega = symplecticform(state1.basis)
    mul!(temp_mat, Omega, V12)              
    mul!(V12, temp_mat, Omega)              
    V12 .*= -1                              
    mul!(temp_vec, V12, xi)                
    arg1 = -0.5 * dot(xi, temp_vec)
    mul!(temp_vec, Omega, μ12)              
    arg2 = 1im * dot(temp_vec, xi)
    return exp(arg1 - arg2)
end

"""
    wignerchar(lc::GaussianLinearCombination, xi::AbstractVector)

Compute Wigner characteristic function of a linear combination including interference.
"""
function wignerchar(lc::GaussianLinearCombination, xi::AbstractVector)
    length(xi) == length(lc.states[1].mean) || throw(ArgumentError(WIGNER_ERROR))
    result = 0.0 + 0.0im
    @inbounds for i in 1:length(lc)
        ci, si = lc[i]
        result += abs2(ci) * wignerchar(si, xi)        
        @inbounds for j in (i+1):length(lc)
            cj, sj = lc[j]
            cross_term = 2 * real(conj(ci) * cj * cross_wignerchar(si, sj, xi))
            result += cross_term
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

"""
    _is_gpu_array(x::AbstractArray)

Check if an array is a GPU array. Extended by CUDA extension.
"""
function _is_gpu_array(x::AbstractArray)
    return false  
end

"""
    _device_aware_vacuum(lc::GaussianLinearCombination)

Create a vacuum state on the same device as the linear combination's states.
"""
function _device_aware_vacuum(lc::GaussianLinearCombination)
    vac = vacuumstate(lc.basis, ħ = lc.ħ)
    if !isempty(lc.states) && _is_gpu_array(lc.states[1].mean)
        T = real(eltype(lc.states[1].mean))
        return gpu(vac, precision = T)
    else
        return vac
    end
end

"""
    _create_device_array(data, reference_array)

Create array with proper device placement for coefficients.
"""
function _create_device_array(data, reference_array)
    T = eltype(data)
    return Vector{T}(Array(data))  
end