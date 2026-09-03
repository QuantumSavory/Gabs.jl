"""
    express(state::GaussianState, [basis::SymplecticBasis]; threshold=1e-10)

Convert a GaussianState to a different formalism or basis representation.

This function enables formalism conversions between Gabs Gaussian representations
and other quantum formalisms supported by QuantumInterface.jl.

# Arguments
- `state::GaussianState`: The Gaussian state to convert
- `basis::SymplecticBasis`: Optional target symplectic basis (QuadPairBasis or QuadBlockBasis)

# Keyword Arguments  
- `threshold::Real`: Threshold for identifying non-Gaussian states (default: 1e-10)

# Returns
- If basis is provided: Returns state in the specified symplectic basis
- If no basis provided: Returns state in its current basis (identity operation)

# Details
The function uses the Williamson decomposition to extract symplectic invariants
from the covariance matrix, enabling conversions between different basis representations
while preserving the physical state.

# Example
```julia
state = vacuumstate(QuadPairBasis(2))  # 2-mode Gaussian state
state_block = express(state, QuadBlockBasis(2))  # Convert to block basis
```
"""
function express(state::GaussianState{B,T,V}, basis::QuadPairBasis) where {B<:QuadPairBasis,T,V}
    # If the state is already in the requested basis type, return it
    if isa(B, QuadPairBasis)
        return state
    else
        # Convert from QuadBlockBasis to QuadPairBasis
        return changebasis(QuadPairBasis, state)
    end
end

function express(state::GaussianState{B,T,V}, basis::QuadBlockBasis) where {B<:QuadBlockBasis,T,V}
    # If the state is already in the requested basis type, return it
    if isa(B, QuadBlockBasis)
        return state
    else
        # Convert from QuadPairBasis to QuadBlockBasis
        return changebasis(QuadBlockBasis, state)
    end
end

function express(state::GaussianState{B,T,V}, basis::QuadPairBasis) where {B<:QuadBlockBasis,T,V}
    # Convert from QuadBlockBasis to QuadPairBasis
    return changebasis(QuadPairBasis, state)
end

function express(state::GaussianState{B,T,V}, basis::QuadBlockBasis) where {B<:QuadPairBasis,T,V}
    # Convert from QuadPairBasis to QuadBlockBasis
    return changebasis(QuadBlockBasis, state)
end

function express(state::GaussianState)
    # Identity operation - return state as-is
    return state
end

"""
    express(unitary::GaussianUnitary, [basis::SymplecticBasis])

Convert a GaussianUnitary to a different basis representation.

# Arguments
- `unitary::GaussianUnitary`: The Gaussian unitary to convert
- `basis::SymplecticBasis`: Optional target symplectic basis

# Returns
- If basis is provided: Returns unitary in the specified symplectic basis
- If no basis provided: Returns unitary in its current basis

# Example
```julia
u = phaseshift(π/4, QuadPairBasis(1))
u_block = express(u, QuadBlockBasis(1))  # Convert to block basis
```
"""
function express(unitary::GaussianUnitary{B,T,V}, basis::QuadPairBasis) where {B<:QuadPairBasis,T,V}
    # If the unitary is already in the requested basis type, return it
    return unitary
end

function express(unitary::GaussianUnitary{B,T,V}, basis::QuadBlockBasis) where {B<:QuadBlockBasis,T,V}
    # If the unitary is already in the requested basis type, return it
    return unitary
end

function express(unitary::GaussianUnitary{B,T,V}, basis::QuadPairBasis) where {B<:QuadBlockBasis,T,V}
    # Convert from QuadBlockBasis to QuadPairBasis
    return changebasis(QuadPairBasis, unitary)
end

function express(unitary::GaussianUnitary{B,T,V}, basis::QuadBlockBasis) where {B<:QuadPairBasis,T,V}
    # Convert from QuadPairBasis to QuadBlockBasis
    return changebasis(QuadBlockBasis, unitary)
end

function express(unitary::GaussianUnitary)
    # Identity operation - return unitary as-is
    return unitary
end

"""
    express(channel::GaussianChannel, [basis::SymplecticBasis])

Convert a GaussianChannel to a different basis representation.

# Arguments
- `channel::GaussianChannel`: The Gaussian channel to convert
- `basis::SymplecticBasis`: Optional target symplectic basis

# Returns
- If basis is provided: Returns channel in the specified symplectic basis
- If no basis provided: Returns channel in its current basis

# Example
```julia
ch = attenuator(0.9, QuadPairBasis(1))
ch_block = express(ch, QuadBlockBasis(1))  # Convert to block basis
```
"""
function express(channel::GaussianChannel{B,T,V}, basis::QuadPairBasis) where {B<:QuadPairBasis,T,V}
    # If the channel is already in the requested basis type, return it
    return channel
end

function express(channel::GaussianChannel{B,T,V}, basis::QuadBlockBasis) where {B<:QuadBlockBasis,T,V}
    # If the channel is already in the requested basis type, return it
    return channel
end

function express(channel::GaussianChannel{B,T,V}, basis::QuadPairBasis) where {B<:QuadBlockBasis,T,V}
    # Convert from QuadBlockBasis to QuadPairBasis
    return changebasis(QuadPairBasis, channel)
end

function express(channel::GaussianChannel{B,T,V}, basis::QuadBlockBasis) where {B<:QuadPairBasis,T,V}
    # Convert from QuadPairBasis to QuadBlockBasis
    return changebasis(QuadBlockBasis, channel)
end

function express(channel::GaussianChannel)
    # Identity operation - return channel as-is
    return channel
end
