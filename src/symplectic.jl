abstract type SymplecticBasis{N} end

nmodes(x::SymplecticBasis) = x.nmodes

"""
Defines a symplectic basis for a bosonic system of size `nmodes` in which
the quadrature field operators are arranged pairwise.
"""
struct QuadPairBasis{N} <: SymplecticBasis{N}
    nmodes::N
end

"""
Defines a symplectic basis for a bosonic system of size `nmodes` in which
the quadrature field operators are arranged blockwise.
"""
struct QuadBlockBasis{N} <: SymplecticBasis{N}
    nmodes::N
end

function Base.show(io::IO, x::SymplecticBasis)
    print(io, "$(nameof(typeof(x)))($(x.nmodes))")
end
function Base.:(*)(n::N, basis::R) where {N<:Number,R<:SymplecticBasis}
    R(n*basis.nmodes)
end

"""
    directsum(basis1::SymplecticBasis, basis2::SymplecticBasis)

Compute the direct sum of symplectic bases.
"""
function directsum(basis1::R, basis2::R) where {R<:SymplecticBasis}
    R(basis1.nmodes + basis2.nmodes)
end

"""
    symplecticform([T = Matrix{Float64},] basis::SymplecticBasis)

Compute the symplectic form matrix of size 2N x 2N corresponding to `basis`.
"""
function symplecticform(basis::QuadPairBasis{N}) where {N<:Int}
    nmodes = basis.nmodes
    Omega = zeros(2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(nmodes)
        Omega[2*i-1, 2*i] = 1.0
        Omega[2*i, 2*i-1] = -1.0
    end
    return Omega
end
function symplecticform(basis::QuadBlockBasis{N}) where {N<:Int}
    nmodes = basis.nmodes
    Omega = zeros(2*nmodes, 2*nmodes)
    @inbounds for i in 1:nmodes, j in nmodes:2*nmodes
        if isequal(i, j-nmodes)
            Omega[i,j] = 1.0
        end
    end
    @inbounds for i in nmodes:2*nmodes, j in 1:nmodes
        if isequal(i-nmodes,j)
            Omega[i, j] = -1.0
        end
    end
    return Omega
end
symplecticform(::Type{T}, basis::SymplecticBasis{N}) where {T, N<:Int} = T(symplecticform(basis))

"""
    _symplecticform(basis, x)

Symplectic form of `basis` in the same array type and element type as `x`.

`symplecticform(basis)` always returns a CPU `Matrix{Float64}`, which cannot be
combined with a matrix living on another device. Building the form alongside the
data it multiplies keeps the whole expression on one device.
"""
_symplecticform(basis::SymplecticBasis, ::AbstractArray) = symplecticform(basis)

"""
    _complexform(basis, x)

Complex symplectic form of `basis` on the same backend as `x`, for the
positive-semidefiniteness checks in `isgaussian`.
"""
_complexform(basis::SymplecticBasis, ::AbstractArray) = symplecticform(Matrix{ComplexF64}, basis)

"""
    _quadindices(basis, modes)

Row/column indices of the quadratures belonging to `modes`, in the layout of
`basis`: `[q₁,p₁,q₂,p₂,…]` for `QuadPairBasis` and `[q₁,…,qₙ,p₁,…,pₙ]` for
`QuadBlockBasis`.

Selecting a set of modes is the same gather in both layouts once the mode
indices are mapped through here, which lets `ptrace` and the indexed `apply!`
share one code path across array backends.
"""
function _quadindices(::QuadPairBasis, modes)
    idx = Vector{Int}(undef, 2*length(modes))
    @inbounds for (k, i) in enumerate(modes)
        idx[2k-1] = 2i - 1
        idx[2k]   = 2i
    end
    return idx
end
function _quadindices(basis::QuadBlockBasis, modes)
    l = length(modes)
    n = basis.nmodes
    idx = Vector{Int}(undef, 2l)
    @inbounds for (k, i) in enumerate(modes)
        idx[k]   = i
        idx[k+l] = i + n
    end
    return idx
end

"""
    _basisperm(::Type{QuadBlockBasis}, nmodes)
    _basisperm(::Type{QuadPairBasis}, nmodes)

Permutation taking a vector indexed in the source layout to the target layout,
for `nmodes` modes. `x[_basisperm(B, n)]` reorders a mean or displacement, and
`A[p, p]` reorders a covariance, symplectic, transform or noise matrix.

Expressing the relabelling as one gather keeps `changebasis` free of per-element
writes, so it runs on any array backend.
"""
function _basisperm(::Type{<:QuadBlockBasis}, nmodes::Int)
    # target [q₁,…,qₙ,p₁,…,pₙ] reads source [q₁,p₁,…] at 1,3,5,… then 2,4,6,…
    return vcat(1:2:2*nmodes, 2:2:2*nmodes)
end
function _basisperm(::Type{<:QuadPairBasis}, nmodes::Int)
    # target [q₁,p₁,q₂,p₂,…] reads source [q₁,…,qₙ,p₁,…,pₙ]
    p = Vector{Int}(undef, 2*nmodes)
    @inbounds for i in Base.OneTo(nmodes)
        p[2i-1] = i
        p[2i]   = i + nmodes
    end
    return p
end

"""
    _like(x, A)

Copy of the host matrix `A` in the array type and element type of `x`.

Auxiliary matrices such as a generaldyne projection or a homodyne squeezing term
are assembled on the host from scalars; they have to be moved onto the moments'
backend before they can be combined with them.
"""
_like(::AbstractArray, A::AbstractMatrix) = A

"""
    _permute(x, p)
    _permutesquare(A, p)

`x[p]` and `A[p, p]`.

The generic definitions are the indexing expressions themselves, so any array
backend is covered. A host `Array` gets a version that walks the result in column order instead,
which the relabelling in `changebasis` is dense enough to notice. The bound is
`Array` rather than `StridedArray` on purpose: a `CuArray` is strided too, and
indexing one element at a time is exactly what it must not do.
"""
_permute(x, p) = x[p]
_permutesquare(A, p) = A[p, p]

function _permute(x::Array, p)
    out = similar(x)
    @inbounds for (i, pi) in enumerate(p)
        out[i] = x[pi]
    end
    return out
end
function _permutesquare(A::Array, p)
    n = length(p)
    out = similar(A, n, n)
    @inbounds for jj in Base.OneTo(n)
        col = @view A[:, p[jj]]
        for ii in Base.OneTo(n)
            out[ii, jj] = col[p[ii]]
        end
    end
    return out
end
