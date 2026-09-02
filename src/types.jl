"""
Defines a Gaussian state for an N-mode bosonic system over a 2N-dimensional phase space.

## Fields

- `basis`: Symplectic basis for Gaussian state.
- `mean`: The mean vector of length 2N.
- `covar`: The covariance matrix of size 2N x 2N.
- `ħ = 2`: Reduced Planck's constant.

## Example

```jldoctest
julia> coherentstate(QuadPairBasis(1), 1.0+im)
GaussianState for 1 mode.
  symplectic basis: QuadPairBasis
mean: 2-element Vector{Float64}:
 2.0
 2.0
covariance: 2×2 Matrix{Float64}:
 1.0  0.0
 0.0  1.0
```
"""
@kwdef struct GaussianState{B<:SymplecticBasis,M,V} <: StateVector{M,V}
    basis::B
    mean::M
    covar::V
    ħ::Number = 2
    function GaussianState(b::B, m::M, v::V; ħ::Number = 2) where {B,M,V}
        all(size(v) .== length(m) .== 2*(b.nmodes)) || throw(DimensionMismatch(STATE_ERROR))
        return new{B,M,V}(b, m, v, ħ)
    end
end

Base.:(==)(x::GaussianState, y::GaussianState) = x.basis == y.basis && x.mean == y.mean && x.covar == y.covar && x.ħ == y.ħ
Base.isapprox(x::GaussianState, y::GaussianState; kwargs...) = x.basis == y.basis && isapprox(x.mean, y.mean; kwargs...) && isapprox(x.covar, y.covar; kwargs...) && x.ħ == y.ħ
function Base.show(io::IO, mime::MIME"text/plain", x::GaussianState)
    Base.summary(io, x)
    print(io, "\n  symplectic basis: ")
    printstyled(io, "$(nameof(typeof(x.basis)))"; color = :blue)
    print(io, "\nmean: ")
    Base.show(io, mime, x.mean)
    print(io, "\ncovariance: ")
    Base.show(io, mime, x.covar)
end
Base.copy(x::GaussianState) = GaussianState(x.basis, copy(x.mean), copy(x.covar); ħ = x.ħ)
nmodes(x::GaussianState) = nmodes(x.basis)

"""
Defines a Gaussian unitary for an N-mode bosonic system over a 2N-dimensional phase space.

## Fields

- `basis`: Symplectic basis for Gaussian unitary.
- `disp`: The displacement vector of length 2N.
- `symplectic`: The symplectic matrix of size 2N x 2N.
- `ħ = 2`: Reduced Planck's constant.

## Mathematical description of a Gaussian unitary

An `N`-mode Gaussian unitary, is a unitary
operator characterized by a displacement vector `d` of length `2N` and symplectic
matrix `S` of size `2N x 2N`, such that its action on a Gaussian state
results in a Gaussian state. More specifically, a Gaussian unitary transformation on a
Gaussian state is described by its maps on
the statistical moments `x̄` and `V` of the Gaussian state: `x̄ → Sx̄ + d` and `V → SVSᵀ`.

## Example

```jldoctest
julia> displace(QuadPairBasis(1), 1.0+im)
GaussianUnitary for 1 mode.
  symplectic basis: QuadPairBasis
displacement: 2-element Vector{Float64}:
 2.0
 2.0
symplectic: 2×2 Matrix{Float64}:
 1.0  0.0
 0.0  1.0
```
"""
@kwdef struct GaussianUnitary{B<:SymplecticBasis,D,S} <: AbstractOperator{D,S}
    basis::B
    disp::D
    symplectic::S
    ħ::Number = 2
    function GaussianUnitary(b::B, d::D, s::S; ħ::Number = 2) where {B,D,S}
        all(size(s) .== length(d) .== 2*(b.nmodes)) || throw(DimensionMismatch(UNITARY_ERROR))
        return new{B,D,S}(b, d, s, ħ)
    end
end

Base.:(==)(x::GaussianUnitary, y::GaussianUnitary) = x.basis == y.basis && x.disp == y.disp && x.symplectic == y.symplectic && x.ħ == y.ħ
Base.isapprox(x::GaussianUnitary, y::GaussianUnitary; kwargs...) = x.basis == y.basis && isapprox(x.disp, y.disp; kwargs...) && isapprox(x.symplectic, y.symplectic; kwargs...) && x.ħ == y.ħ
function Base.show(io::IO, mime::MIME"text/plain", x::GaussianUnitary)
    Base.summary(io, x)
    print(io, "\n  symplectic basis: ")
    printstyled(io, "$(nameof(typeof(x.basis)))"; color = :blue)
    print(io, "\ndisplacement: ")
    Base.show(io, mime, x.disp)
    print(io, "\nsymplectic: ")
    Base.show(io, mime, x.symplectic)
end
Base.copy(x::GaussianUnitary) = GaussianUnitary(x.basis, copy(x.disp), copy(x.symplectic); ħ = x.ħ)
nmodes(x::GaussianUnitary) = nmodes(x.basis)

function Base.:(*)(op1::GaussianUnitary, op2::GaussianUnitary)
    op1.basis == op2.basis || throw(DimensionMismatch(ACTION_ERROR))
    op1.ħ == op2.ħ || throw(ArgumentError(HBAR_ERROR))
    d1, S1 = op1.disp, op1.symplectic
    d2, S2 = op2.disp, op2.symplectic
    return GaussianUnitary(op1.basis, S1 * d2 .+ d1, S1 * S2; ħ = op1.ħ)
end
function Base.:(*)(op::GaussianUnitary, state::GaussianState)
    op.basis == state.basis || throw(DimensionMismatch(ACTION_ERROR))
    op.ħ == state.ħ || throw(ArgumentError(HBAR_ERROR))
    d, S, = op.disp, op.symplectic
    mean′ = S * state.mean .+ d
    covar′ = S * state.covar * transpose(S)
    return GaussianState(state.basis, mean′, covar′; ħ = state.ħ)
end
"""
    apply!(state::GaussianState, op::GaussianUnitary)

In-place application of a Gaussian unitary `op` on a Gaussian state `state`.
"""
function apply!(state::GaussianState, op::GaussianUnitary)
    op.basis == state.basis || throw(DimensionMismatch(ACTION_ERROR))
    op.ħ == state.ħ || throw(ArgumentError(HBAR_ERROR))
    d, S = op.disp, op.symplectic
    state.mean .= S * state.mean .+ d
    state.covar .= S * state.covar * transpose(S)
    return state
end
"""
    apply!(state::GaussianState, index::Int, op::GaussianUnitary)
    apply!(state::GaussianState, indices::AbstractVector{<:Int}, op::GaussianUnitary)

In-place application of a Gaussian unitary `op` on the mode index or indices of a
Gaussian state `state`.
"""
function apply!(state::GaussianState, index::Int, op::GaussianUnitary)
    return apply!(state, [index], op)
end
function apply!(
    state::GaussianState{B,M,V},
    indices::AbstractVector{<:Int},
    op::GaussianUnitary,
) where {B<:QuadPairBasis,M,V}
    typeof(op.basis) == typeof(state.basis) || throw(DimensionMismatch(ACTION_ERROR))
    op.ħ == state.ħ || throw(ArgumentError(HBAR_ERROR))
    length(indices) ≤ state.basis.nmodes || throw(ArgumentError(INDEX_ERROR))
    quad_indices = Vector{Int}(undef, 2length(indices))
    @inbounds for (k, i) in enumerate(indices)
        quad_indices[2k-1] = 2i - 1
        quad_indices[2k]   = 2i
    end
    d, S = op.disp, op.symplectic
    m = length(quad_indices)
    n = size(state.covar, 1)
    mean_sub = @view state.mean[quad_indices]
    covar_row = @view state.covar[quad_indices, :]
    covar_col = @view state.covar[:, quad_indices]
    # single scratch buffer, reused across the three products (reshaped for the column update)
    buf = similar(state.covar, m, n)
    buf_vec = @view buf[1:m]
    # x̄[q] ← S x̄[q] + d
    mul!(buf_vec, S, mean_sub)
    mean_sub .= buf_vec .+ d
    # V[q,:] ← S V[q,:]
    mul!(buf, S, covar_row)
    covar_row .= buf
    # V[:,q] ← V[:,q] Sᵀ (reads the just-updated V[q,q] block)
    buf_col = reshape(buf, n, m)
    mul!(buf_col, covar_col, transpose(S))
    covar_col .= buf_col
    return state
end
function apply!(
    state::GaussianState{B,M,V},
    indices::AbstractVector{<:Int},
    op::GaussianUnitary,
) where {B<:QuadBlockBasis,M,V}
    typeof(op.basis) == typeof(state.basis) || throw(DimensionMismatch(ACTION_ERROR))
    op.ħ == state.ħ || throw(ArgumentError(HBAR_ERROR))
    length(indices) ≤ state.basis.nmodes || throw(ArgumentError(INDEX_ERROR))
    l = length(indices)
    quad_indices = Vector{Int}(undef, 2l)
    @inbounds for (k, i) in enumerate(indices)
        quad_indices[k]   = i
        quad_indices[k+l] = i + state.basis.nmodes
    end
    d, S = op.disp, op.symplectic
    m = length(quad_indices)
    n = size(state.covar, 1)
    mean_sub = @view state.mean[quad_indices]
    covar_row = @view state.covar[quad_indices, :]
    covar_col = @view state.covar[:, quad_indices]
    # single scratch buffer, reused across the three products (reshaped for the column update)
    buf = similar(state.covar, m, n)
    buf_vec = @view buf[1:m]
    # x̄[q] ← S x̄[q] + d
    mul!(buf_vec, S, mean_sub)
    mean_sub .= buf_vec .+ d
    # V[q,:] ← S V[q,:]
    mul!(buf, S, covar_row)
    covar_row .= buf
    # V[:,q] ← V[:,q] Sᵀ (reads the just-updated V[q,q] block)
    buf_col = reshape(buf, n, m)
    mul!(buf_col, covar_col, transpose(S))
    covar_col .= buf_col
    return state
end

Base.@deprecate(
    apply!(
        state::GaussianState,
        op::GaussianUnitary,
        indices::AbstractVector{<:Int},
    ),
    apply!(state, indices, op)
)

"""
Defines a Gaussian channel for an N-mode bosonic system over a 2N-dimensional phase space.

## Fields

- `basis`: Symplectic representation for Gaussian channel.
- `disp`: The displacement vector of length 2N.
- `transform`: The transformation matrix of size 2N x 2N.
- `noise`: The noise matrix of size 2N x 2N.
- `ħ = 2`: Reduced Planck's constant.

## Mathematical description of a Gaussian channel

An `N`-mode Gaussian channel is an
operator characterized by a displacement vector `d` of length `2N`, as well as
a transformation matrix `T` and noise matrix `N` of size `2N x 2N`,
such that its action on a Gaussian state results in a Gaussian state. More specifically, a Gaussian
channel action on a Gaussian state is described by its maps on
the statistical moments `x̄` and `V` of the Gaussian state: `x̄ → Tx̄ + d` and `V → TVTᵀ + N`.

## Example

```jldoctest
julia> noise = [1.0 -3.0; 4.0 2.0];

julia> displace(QuadPairBasis(1), 1.0+im, noise)
GaussianChannel for 1 mode.
  symplectic basis: QuadPairBasis
displacement: 2-element Vector{Float64}:
 2.0
 2.0
transform: 2×2 Matrix{Float64}:
 1.0  0.0
 0.0  1.0
noise: 2×2 Matrix{Float64}:
 1.0  -3.0
 4.0   2.0
```
"""
@kwdef struct GaussianChannel{B<:SymplecticBasis,D,T} <: AbstractOperator{D,T}
    basis::B
    disp::D
    transform::T
    noise::T
    ħ::Number = 2
    function GaussianChannel(b::B, d::D, t::T, n::T; ħ::Number = 2) where {B,D,T}
        all(length(d) .== size(t) .== size(n) .== 2*(b.nmodes)) || throw(DimensionMismatch(CHANNEL_ERROR))
        return new{B,D,T}(b, d, t, n, ħ)
    end
end

Base.:(==)(x::GaussianChannel, y::GaussianChannel) = x.basis == y.basis && x.disp == y.disp && x.transform == y.transform && x.noise == y.noise && x.ħ == y.ħ
Base.isapprox(x::GaussianChannel, y::GaussianChannel; kwargs...) = x.basis == y.basis && isapprox(x.disp, y.disp; kwargs...) && isapprox(x.transform, y.transform; kwargs...) && isapprox(x.noise, y.noise; kwargs...) && x.ħ == y.ħ
function Base.show(io::IO, mime::MIME"text/plain", x::GaussianChannel)
    Base.summary(io, x)
    print(io, "\n  symplectic basis: ")
    printstyled(io, "$(nameof(typeof(x.basis)))"; color = :blue)
    print(io, "\ndisplacement: ")
    Base.show(io, mime, x.disp)
    print(io, "\ntransform: ")
    Base.show(io, mime, x.transform)
    print(io, "\nnoise: ")
    Base.show(io, mime, x.noise)
end
Base.copy(x::GaussianChannel) = GaussianChannel(x.basis, copy(x.disp), copy(x.transform), copy(x.noise); ħ = x.ħ)
nmodes(x::GaussianChannel) = nmodes(x.basis)

function Base.:(*)(op::GaussianChannel, state::GaussianState)
    op.basis == state.basis || throw(DimensionMismatch(ACTION_ERROR))
    op.ħ == state.ħ || throw(ArgumentError(HBAR_ERROR))
    d, T, N = op.disp, op.transform, op.noise
    mean′ = T * state.mean .+ d
    covar′ = T * state.covar * transpose(T) .+ N
    return GaussianState(state.basis, mean′, covar′; ħ = state.ħ)
end
"""
    apply!(state::GaussianState, op::GaussianChannel)
    apply!(state::GaussianState, index::Int, op::GaussianChannel)
    apply!(state::GaussianState, indices::AbstractVector{<:Int}, op::GaussianChannel)

In-place application of a Gaussian channel `op` on all or selected modes of a
Gaussian state `state`.
"""
function apply!(state::GaussianState, op::GaussianChannel)
    op.basis == state.basis || throw(DimensionMismatch(ACTION_ERROR))
    op.ħ == state.ħ || throw(ArgumentError(HBAR_ERROR))
    d, T, N = op.disp, op.transform, op.noise
    state.mean .= T * state.mean .+ d
    state.covar .= T * state.covar * transpose(T) .+ N
    return state
end
function apply!(state::GaussianState, index::Int, op::GaussianChannel)
    return apply!(state, [index], op)
end
function apply!(
    state::GaussianState{B,M,V},
    indices::AbstractVector{<:Int},
    op::GaussianChannel,
) where {B<:QuadPairBasis,M,V}
    typeof(op.basis) == typeof(state.basis) || throw(DimensionMismatch(ACTION_ERROR))
    op.ħ == state.ħ || throw(ArgumentError(HBAR_ERROR))
    length(indices) ≤ state.basis.nmodes || throw(ArgumentError(INDEX_ERROR))
    quad_indices = Vector{Int}(undef, 2length(indices))
    @inbounds for (k, i) in enumerate(indices)
        quad_indices[2k-1] = 2i - 1
        quad_indices[2k]   = 2i
    end
    d, T, N = op.disp, op.transform, op.noise
    m = length(quad_indices)
    n = size(state.covar, 1)
    mean_sub = @view state.mean[quad_indices]
    covar_row = @view state.covar[quad_indices, :]
    covar_col = @view state.covar[:, quad_indices]
    # single scratch buffer, reused across the three products (reshaped for the column update)
    buf = similar(state.covar, m, n)
    buf_vec = @view buf[1:m]
    # x̄[q] ← T x̄[q] + d
    mul!(buf_vec, T, mean_sub)
    mean_sub .= buf_vec .+ d
    # V[q,:] ← T V[q,:]
    mul!(buf, T, covar_row)
    covar_row .= buf
    # V[:,q] ← V[:,q] Tᵀ (reads the just-updated V[q,q] block)
    buf_col = reshape(buf, n, m)
    mul!(buf_col, covar_col, transpose(T))
    covar_col .= buf_col
    # V[q,q] ← V[q,q] + N after both covariance transformations
    covar_sub = @view state.covar[quad_indices, quad_indices]
    covar_sub .+= N
    return state
end
function apply!(
    state::GaussianState{B,M,V},
    indices::AbstractVector{<:Int},
    op::GaussianChannel,
) where {B<:QuadBlockBasis,M,V}
    typeof(op.basis) == typeof(state.basis) || throw(DimensionMismatch(ACTION_ERROR))
    op.ħ == state.ħ || throw(ArgumentError(HBAR_ERROR))
    length(indices) ≤ state.basis.nmodes || throw(ArgumentError(INDEX_ERROR))
    l = length(indices)
    quad_indices = Vector{Int}(undef, 2l)
    @inbounds for (k, i) in enumerate(indices)
        quad_indices[k]   = i
        quad_indices[k+l] = i + state.basis.nmodes
    end
    d, T, N = op.disp, op.transform, op.noise
    m = length(quad_indices)
    n = size(state.covar, 1)
    mean_sub = @view state.mean[quad_indices]
    covar_row = @view state.covar[quad_indices, :]
    covar_col = @view state.covar[:, quad_indices]
    # single scratch buffer, reused across the three products (reshaped for the column update)
    buf = similar(state.covar, m, n)
    buf_vec = @view buf[1:m]
    # x̄[q] ← T x̄[q] + d
    mul!(buf_vec, T, mean_sub)
    mean_sub .= buf_vec .+ d
    # V[q,:] ← T V[q,:]
    mul!(buf, T, covar_row)
    covar_row .= buf
    # V[:,q] ← V[:,q] Tᵀ (reads the just-updated V[q,q] block)
    buf_col = reshape(buf, n, m)
    mul!(buf_col, covar_col, transpose(T))
    covar_col .= buf_col
    # V[q,q] ← V[q,q] + N after both covariance transformations
    covar_sub = @view state.covar[quad_indices, quad_indices]
    covar_sub .+= N
    return state
end

"""
Defines a stellar state for an N-mode bosonic system: a Gaussian unitary applied to a
core state of finite Fock support.

## Fields

- `core`: Fock amplitude tensor with one index per mode.
- `gaussian`: The Gaussian unitary factor, which also supplies `basis` and `ħ`.

## Mathematical description of a stellar state

A stellar state of rank `r` is `|ψ⟩ = U|C⟩`, where `U` is a Gaussian unitary and
`|C⟩ = ∑ₖ c[k] |k⟩` is a normalized superposition of Fock states over occupation
multi-indices `k = (k₁,…,kₙ)`, of total degree `r`. The rank `r` is invariant under
every Gaussian unitary and vanishes exactly on the pure Gaussian states.

The decomposition is not unique: `(|C⟩, U) → (V|C⟩, U V†)` for any passive `V` denotes
the same ray, since only passive unitaries preserve finite Fock support. 
Therefore, in Gabs, equality of `StellarState` objects is equality of the stored data, 
not of the states.

## Example

```jldoctest
julia> fockstate(QuadPairBasis(1), 2)
StellarState for 1 mode.
  symplectic basis: QuadPairBasis
  stellar rank: 2
core: 3-element Vector{ComplexF64}:
 0.0 + 0.0im
 0.0 + 0.0im
 1.0 + 0.0im
displacement: 2-element Vector{Float64}:
 0.0
 0.0
symplectic: 2×2 Matrix{Float64}:
 1.0  0.0
 0.0  1.0
```
"""
struct StellarState{C,G<:GaussianUnitary} <: StateVector{C,G}
    core::C
    gaussian::G
    function StellarState(c::C, g::G) where {C,G<:GaussianUnitary}
        ndims(c) == nmodes(g) || throw(DimensionMismatch(STELLAR_ERROR))
        any(!iszero, c) || throw(ArgumentError(CORE_ERROR))
        return new{C,G}(c, g)
    end
end
function Base.getproperty(x::StellarState, s::Symbol)
    return (s === :basis || s === :ħ) ? getproperty(getfield(x, :gaussian), s) : getfield(x, s)
end
Base.propertynames(::StellarState) = (:basis, :core, :gaussian, :ħ)
Base.:(==)(x::StellarState, y::StellarState) = x.core == y.core && x.gaussian == y.gaussian
Base.isapprox(x::StellarState, y::StellarState; kwargs...) =
    size(x.core) == size(y.core) && isapprox(x.core, y.core; kwargs...) &&
    isapprox(x.gaussian, y.gaussian; kwargs...)
Base.copy(x::StellarState) = StellarState(copy(x.core), copy(x.gaussian))
nmodes(x::StellarState) = nmodes(x.gaussian)

function Base.show(io::IO, mime::MIME"text/plain", x::StellarState)
    Base.summary(io, x)
    print(io, "\n  symplectic basis: ")
    printstyled(io, "$(nameof(typeof(x.basis)))"; color = :blue)
    print(io, "\n  stellar rank: ")
    printstyled(io, "$(stellarrank(x))"; color = :blue)
    print(io, "\ncore: ")
    Base.show(io, mime, x.core)
    print(io, "\ndisplacement: ")
    Base.show(io, mime, x.gaussian.disp)
    print(io, "\nsymplectic: ")
    Base.show(io, mime, x.gaussian.symplectic)
end

function Base.:(*)(op::GaussianUnitary, state::StellarState)
    return StellarState(state.core, op * state.gaussian)
end

"""
    apply!(state::StellarState, op::GaussianUnitary)
    apply!(state::StellarState, indices::AbstractVector, op::GaussianUnitary)
    apply!(state::StellarState, index::Int, op::GaussianUnitary)

In-place application of a Gaussian unitary `op` on a stellar state `state`.
Specify `indices` to define the subspace of the stellar state for unitary application.
"""
function apply!(state::StellarState, op::GaussianUnitary)
    op.basis == state.basis || throw(DimensionMismatch(ACTION_ERROR))
    op.ħ == state.ħ || throw(ArgumentError(HBAR_ERROR))
    d, S = op.disp, op.symplectic
    g = state.gaussian
    g.disp .= S * g.disp .+ d
    g.symplectic .= S * g.symplectic
    return state
end
function apply!(state::StellarState, index::Int, op::GaussianUnitary)
    return apply!(state, [index], op)
end
function apply!(state::StellarState{C,G}, indices::AbstractVector{<:Int}, op::GaussianUnitary) where {C,G<:GaussianUnitary{<:QuadPairBasis}}
    typeof(op.basis) == typeof(state.basis) || throw(DimensionMismatch(ACTION_ERROR))
    op.ħ == state.ħ || throw(ArgumentError(HBAR_ERROR))
    length(indices) ≤ state.basis.nmodes || throw(ArgumentError(INDEX_ERROR))
    quad_indices = Vector{Int}(undef, 2length(indices))
    @inbounds for (k, i) in enumerate(indices)
        quad_indices[2k-1] = 2i - 1
        quad_indices[2k]   = 2i
    end
    return _applygaussian!(state, quad_indices, op)
end
function apply!(state::StellarState{C,G}, indices::AbstractVector{<:Int}, op::GaussianUnitary) where {C,G<:GaussianUnitary{<:QuadBlockBasis}}
    typeof(op.basis) == typeof(state.basis) || throw(DimensionMismatch(ACTION_ERROR))
    op.ħ == state.ħ || throw(ArgumentError(HBAR_ERROR))
    length(indices) ≤ state.basis.nmodes || throw(ArgumentError(INDEX_ERROR))
    l = length(indices)
    quad_indices = Vector{Int}(undef, 2l)
    @inbounds for (k, i) in enumerate(indices)
        quad_indices[k]   = i
        quad_indices[k+l] = i + state.basis.nmodes
    end
    return _applygaussian!(state, quad_indices, op)
end
function _applygaussian!(state::StellarState, quad_indices::Vector{Int}, op::GaussianUnitary)
    d, S = op.disp, op.symplectic
    g = state.gaussian
    m = length(quad_indices)
    n = 2 * state.basis.nmodes
    disp_sub = @view g.disp[quad_indices]
    symp_row = @view g.symplectic[quad_indices, :]
    # single buffer that's reused across the two products
    buf = similar(g.symplectic, m, n)
    buf_vec = @view buf[1:m]
    # d[q] ← S d[q] + d_op
    mul!(buf_vec, S, disp_sub)
    disp_sub .= buf_vec .+ d
    # G[q,:] ← S G[q,:]
    mul!(buf, S, symp_row)
    symp_row .= buf
    return state
end

"""
    stellarrank(x::StellarState)

Total degree of the core tensor, i.e. the stellar rank of the state.
"""
function stellarrank(x::StellarState)
    n = nmodes(x)
    rank = 0
    @inbounds for I in CartesianIndices(x.core)
        iszero(x.core[I]) && continue
        rank = max(rank, sum(Tuple(I)) - n)
    end
    return rank
end
stellarrank(::GaussianState) = 0

"""
    GaussianState(x::StellarState)

Return Gaussian state obtained from applying the Gaussian unitary part of `x`
to a vacuum state.
"""
function GaussianState(x::StellarState)
    iszero(stellarrank(x)) || throw(ArgumentError(RANK_ERROR))
    return x.gaussian * vacuumstate(x.basis; ħ = x.ħ)
end
"""
    StellarState(x::GaussianState; atol = 1e-8)

Rank-zero stellar state of a pure Gaussian state. The Gaussian factor is the positive
symmetric symplectic square root of `(2/ħ)V`, which satisfies `V = (ħ/2)GGᵀ` by
construction; the mean supplies `d`. Inverse of `GaussianState(::StellarState)`.

A Gaussian state admits this decomposition exactly when it is pure, and `(2/ħ)V` is
symplectic exactly then, so the square root is symplectic whenever it exists.
"""
function StellarState(x::GaussianState{B,M,V}; atol::Real = 1e-8) where {B,M,V}
    F = eigen(Symmetric((2/x.ħ) .* x.covar))
    all(>(0), F.values) || throw(ArgumentError(
        lazy"The covariance matrix is not positive definite, so its symmetric square root
        does not exist."))
    G = F.vectors * Diagonal(sqrt.(F.values)) * transpose(F.vectors)
    issymplectic(x.basis, G; atol = atol, rtol = atol) ||
        throw(ArgumentError(STELLAR_PURITY_ERROR))
    core = fill(one(ComplexF64), ntuple(_ -> 1, nmodes(x.basis)))
    return StellarState(core, GaussianUnitary(x.basis, copy(x.mean), G; ħ = x.ħ))
end

purity(x::StellarState) = one(real(eltype(x.core)))
entropy_vn(x::StellarState) = zero(real(eltype(x.core)))

"""
    isgaussian(x::GaussianState)
    isgaussian(x::GaussianUnitary)
    isgaussian(x::GaussianChannel)

Check if `x` satisfies the corresponding Gaussian definition for its type.

## Example

```jldoctest
julia> basis = QuadPairBasis(1);

julia> op = displace(basis, 1.0-im)
GaussianUnitary for 1 mode.
  symplectic basis: QuadPairBasis
displacement: 2-element Vector{Float64}:
  2.0
 -2.0
symplectic: 2×2 Matrix{Float64}:
 1.0  0.0
 0.0  1.0

julia> isgaussian(op)
true
```
"""
function isgaussian(x::GaussianState; atol::R1 = 0, rtol::R2 = atol) where {R1<:Real, R2<:Real}
    covar = x.covar
    basis = x.basis
    form = symplecticform(Matrix{ComplexF64}, basis)
    @. form = im * (x.ħ/2) * form + covar
    eigs = real(eigvals(form))
    return all(i -> ((i >= 0) || isapprox(i, 0.0; atol = atol, rtol = rtol)), eigs)
end
function isgaussian(x::GaussianUnitary; atol::R1 = 0, rtol::R2 = atol) where {R1<:Real, R2<:Real} 
    return issymplectic(x.basis, x.symplectic; atol = atol, rtol = rtol)
end
function isgaussian(x::GaussianChannel; atol::R1 = 0, rtol::R2 = atol) where {R1<:Real, R2<:Real} 
    transform, noise = x.transform, x.noise
    basis = x.basis
    form = symplecticform(Matrix{ComplexF64}, basis)
    prod = transform * form * transform'
    @. form = noise + im*form - im*prod
    eigs = real(eigvals(form))
    return all(i -> ((i >= 0) || isapprox(i, 0.0; atol = atol, rtol = rtol)), eigs)
end
function isgaussian(x::StellarState; atol::R1 = 0, rtol::R2 = atol) where {R1<:Real,R2<:Real}
    return iszero(stellarrank(x)) && isgaussian(x.gaussian; atol = atol, rtol = rtol)
end


function Base.summary(io::IO, x::Union{GaussianState,GaussianUnitary,GaussianChannel,StellarState})
    printstyled(io, nameof(typeof(x)); color=:blue)
    basis = x.basis
    modenum = basis.nmodes
    if isone(modenum)
        print(io, " for $(modenum) mode.")
    else
        print(io, " for $(modenum) modes.")
    end
end