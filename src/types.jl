"""
Defines a Gaussian state for an N-mode bosonic system over a 2N-dimensional phase space.

## Fields

- `basis`: Symplectic basis for Gaussian state.
- `mean`: The mean vector of length 2N.
- `covar`: The covariance matrix of size 2N x 2N.

## Mathematical description of a Gaussian state

An ``N``-mode Gaussian state, ``\\hat{\\rho}(\\mathbf{\\bar{x}}, \\mathbf{V})``, is a density
operator characterized by two statistical moments: a mean vector ``\\mathbf{\\bar{x}}`` of
length ``2N`` and covariance matrix ``\\mathbf{V}`` of size ``2N\\times 2N``. By definition,
the Wigner representation of a Gaussian state is a Gaussian function.

## Example

```jldoctest
julia> coherentstate(QuadPairBasis(1), 1.0+im)
GaussianState for 1 mode in QuadPairBasis representation.
mean: 2-element Vector{Float64}:
 1.4142135623730951
 1.4142135623730951
covariance: 2×2 Matrix{Float64}:
 1.0  0.0
 0.0  1.0
```
"""
struct GaussianState{B<:SymplecticBasis,M,V} <: StateVector{M,V}
    basis::B
    mean::M
    covar::V
    function GaussianState(b::B, m::M, v::V) where {B,M,V}
        all(size(v) .== length(m) .== 2*(b.nmodes)) || throw(DimensionMismatch(STATE_ERROR))
        return new{B,M,V}(b, m, v)
    end
end

Base.:(==)(x::GaussianState, y::GaussianState) = x.basis == y.basis && x.mean == y.mean && x.covar == y.covar
Base.isapprox(x::GaussianState, y::GaussianState) = x.basis == y.basis && isapprox(x.mean,y.mean) && isapprox(x.covar,y.covar)
function Base.show(io::IO, mime::MIME"text/plain", x::GaussianState)
    Base.summary(io, x)
    print(io, "\nmean: ")
    Base.show(io, mime, x.mean)
    print(io, "\ncovariance: ")
    Base.show(io, mime, x.covar)
end


"""
Defines a Gaussian unitary for an N-mode bosonic system over a 2N-dimensional phase space.

## Fields

- `basis`: Symplectic basis for Gaussian unitary.
- `disp`: The displacement vector of length 2N.
- `symplectic`: The symplectic matrix of size 2N x 2N.

## Mathematical description of a Gaussian unitary

An ``N``-mode Gaussian unitary, ``U(\\mathbf{d}, \\mathbf{S})``, is a unitary
operator characterized by a displacement vector ``\\mathbf{d}`` of length ``2N`` and symplectic
matrix ``\\mathbf{S}`` of size ``2N\\times 2N``, such that its action on a Gaussian state
results in a Gaussian state. More specifically, a Gaussian unitary transformation on a
Gaussian state ``\\hat{\\rho}(\\mathbf{\\bar{x}}, \\mathbf{V})`` is described by its maps on
the statistical moments of the Gaussian state:

```math
\\mathbf{\\bar{x}} \\to \\mathbf{S} \\mathbf{\\bar{x}} + \\mathbf{d}, \\quad
\\mathbf{V} \\to \\mathbf{S} \\mathbf{V} \\mathbf{S}^{\\text{T}}.
```

## Example

```jldoctest
julia> displace(QuadPairBasis(1), 1.0+im)
GaussianUnitary for 1 mode in QuadPairBasis representation.
displacement: 2-element Vector{Float64}:
 1.4142135623730951
 1.4142135623730951
symplectic: 2×2 Matrix{Float64}:
 1.0  0.0
 0.0  1.0
```
"""
struct GaussianUnitary{B<:SymplecticBasis,D,S} <: AbstractOperator{D,S}
    basis::B
    disp::D
    symplectic::S
    function GaussianUnitary(b::B, d::D, s::S) where {B,D,S}
        all(size(s) .== length(d) .== 2*(b.nmodes)) || throw(DimensionMismatch(UNITARY_ERROR))
        return new{B,D,S}(b, d, s)
    end
end

Base.:(==)(x::GaussianUnitary, y::GaussianUnitary) = x.basis == y.basis && x.disp == y.disp && x.symplectic == y.symplectic
Base.isapprox(x::GaussianUnitary, y::GaussianUnitary) = x.basis == y.basis && isapprox(x.disp, y.disp) && isapprox(x.symplectic, y.symplectic)
function Base.show(io::IO, mime::MIME"text/plain", x::GaussianUnitary)
    Base.summary(io, x)
    print(io, "\ndisplacement: ")
    Base.show(io, mime, x.disp)
    print(io, "\nsymplectic: ")
    Base.show(io, mime, x.symplectic)
end

function Base.:(*)(op::GaussianUnitary, state::GaussianState)
    op.basis == state.basis || throw(DimensionMismatch(ACTION_ERROR))
    d, S, = op.disp, op.symplectic
    mean′ = S * state.mean .+ d
    covar′ = S * state.covar * transpose(S)
    return GaussianState(state.basis, mean′, covar′)
end
function apply!(state::GaussianState, op::GaussianUnitary)
    op.basis == state.basis || throw(DimensionMismatch(ACTION_ERROR))
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

## Mathematical description of a Gaussian channel

An ``N``-mode Gaussian channel, ``G(\\mathbf{d}, \\mathbf{T}, \\mathbf{N})``, is an
operator characterized by a displacement vector ``\\mathbf{d}`` of length ``2N``, as well as
a transformation matrix ``\\mathbf{T}`` and noise matrix ``\\mathbf{N}`` of size ``2N\\times 2N``,
such that its action on a Gaussian state results in a Gaussian state. More specifically, a Gaussian
channel action on a Gaussian state ``\\hat{\\rho}(\\mathbf{\\bar{x}}, \\mathbf{V})`` is
described by its maps on the statistical moments of the Gaussian state:

```math
\\mathbf{\\bar{x}} \\to \\mathbf{T} \\mathbf{\\bar{x}} + \\mathbf{d}, \\quad
\\mathbf{V} \\to \\mathbf{T} \\mathbf{V} \\mathbf{T}^{\\text{T}} + \\mathbf{N}.
```

## Example

```jldoctest
julia> noise = [1.0 -3.0; 4.0 2.0];

julia> displace(QuadPairBasis(1), 1.0+im, noise)
GaussianChannel for 1 mode in QuadPairBasis representation.
displacement: 2-element Vector{Float64}:
 1.4142135623730951
 1.4142135623730951
transform: 2×2 Matrix{Float64}:
 1.0  0.0
 0.0  1.0
noise: 2×2 Matrix{Float64}:
 1.0  -3.0
 4.0   2.0
```
"""
struct GaussianChannel{B<:SymplecticBasis,D,T} <: AbstractOperator{D,T}
    basis::B
    disp::D
    transform::T
    noise::T
    function GaussianChannel(b::B, d::D, t::T, n::T) where {B,D,T}
        all(length(d) .== size(t) .== size(n) .== 2*(b.nmodes)) || throw(DimensionMismatch(CHANNEL_ERROR))
        return new{B,D,T}(b, d, t, n)
    end
end

Base.:(==)(x::GaussianChannel, y::GaussianChannel) = x.basis == y.basis && x.disp == y.disp && x.transform == y.transform && x.noise == y.noise
Base.isapprox(x::GaussianChannel, y::GaussianChannel) = x.basis == y.basis && isapprox(x.disp, y.disp) && isapprox(x.transform, y.transform) && isapprox(x.noise, y.noise)
function Base.show(io::IO, mime::MIME"text/plain", x::GaussianChannel)
    Base.summary(io, x)
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
        print(io, " for $(modenum) mode ")
    else
        print(io, " for $(modenum) modes ")
    end
    print(io, "in ")
    printstyled(io, "$(nameof(typeof(basis)))"; color = :blue)
    print(io, " representation.")
end

function Base.:(*)(op::GaussianChannel, state::GaussianState)
    op.basis == state.basis || throw(DimensionMismatch(ACTION_ERROR))
    d, T, N = op.disp, op.transform, op.noise
    mean′ = T * state.mean .+ d
    covar′ = T * state.covar * transpose(T) .+ N
    return GaussianState(state.basis, mean′, covar′)
end
function apply!(state::GaussianState, op::GaussianChannel)
    op.basis == state.basis || throw(DimensionMismatch(ACTION_ERROR))
    d, T, N = op.disp, op.transform, op.noise
    state.mean .= T * state.mean .+ d
    state.covar .= T * state.covar * transpose(T) .+ N
    return state
end