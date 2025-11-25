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
function Base.summary(io::IO, x::Union{GaussianState,GaussianUnitary,GaussianChannel})
    printstyled(io, nameof(typeof(x)); color=:blue)
    basis = x.basis
    modenum = basis.nmodes
    if isone(modenum)
        print(io, " for $(modenum) mode.")
    else
        print(io, " for $(modenum) modes.")
    end
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

In-place application of a Gaussian channel `op` on a Gaussian state `state`.
"""
function apply!(state::GaussianState, op::GaussianChannel)
    op.basis == state.basis || throw(DimensionMismatch(ACTION_ERROR))
    op.ħ == state.ħ || throw(ArgumentError(HBAR_ERROR))
    d, T, N = op.disp, op.transform, op.noise
    state.mean .= T * state.mean .+ d
    state.covar .= T * state.covar * transpose(T) .+ N
    return state
end

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