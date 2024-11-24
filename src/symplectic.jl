abstract type SymplecticRepr{N} end

struct CanonicalForm{N} <: SymplecticRepr{N}
    nmodes::N
end

struct BlockForm{N} <: SymplecticRepr{N}
    nmodes::N
end

function Base.:(*)(n::N, repr2::R) where {N<:Number,R<:SymplecticRepr}
    R(n*repr2.nmodes)
end
function Base.:(+)(repr1::R, repr2::R) where {R<:SymplecticRepr}
    R(repr1.nmodes + repr2.nmodes)
end
function Base.:(-)(repr1::R, repr2::R) where {R<:SymplecticRepr}
    R(repr1.nmodes - repr2.nmodes)
end

"""
    symplecticform([T = Matrix{Float64},] nmodes<:Int)

Compute the symplectic form matrix of size 2N x 2N, where N is given by `nmodes`.
"""
function symplecticform(repr::CanonicalForm{N}) where {N<:Int}
    nmodes = repr.nmodes
    Omega = zeros(2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(nmodes)
        Omega[2*i-1, 2*i] = 1.0
        Omega[2*i, 2*i-1] = -1.0
    end
    return Omega
end
symplecticform(::Type{T}, repr::SymplecticRepr{N}) where {T, N<:Int} = T(symplecticform(repr))

