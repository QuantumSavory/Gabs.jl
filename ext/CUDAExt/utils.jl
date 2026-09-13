# Output container selection, mirroring `ext/StaticArraysExt/utils.jl`.
#
# `tensor` and `ptrace` compute into a plain `Array` and then ask which container
# the result should live in. `promote_type(CuVector{T}, Vector{T})` is `Any`, so
# the mixed pairs need their own methods rather than falling through to the
# generic one. The output argument is annotated so the three-argument form and
# the `(type, output, size)` form cannot overlap.

Base.@propagate_inbounds function _promote_output_vector(::Type{T1}, ::Type{T2}, vec_out::AbstractVector) where {T1<:CuVector,T2<:CuVector}
    return CuVector{promote_type(eltype(T1), eltype(T2))}(vec_out)
end
Base.@propagate_inbounds function _promote_output_vector(::Type{T1}, ::Type{T2}, vec_out::AbstractVector) where {T1<:CuVector,T2<:AbstractVector}
    return CuVector{promote_type(eltype(T1), eltype(T2))}(vec_out)
end
Base.@propagate_inbounds function _promote_output_vector(::Type{T1}, ::Type{T2}, vec_out::AbstractVector) where {T1<:AbstractVector,T2<:CuVector}
    return CuVector{promote_type(eltype(T1), eltype(T2))}(vec_out)
end
Base.@propagate_inbounds function _promote_output_vector(::Type{T}, vec_out::AbstractVector, vec_length::Int) where {T<:CuVector}
    return CuVector{eltype(T)}(vec_out)
end

Base.@propagate_inbounds function _promote_output_matrix(::Type{T1}, ::Type{T2}, mat_out::AbstractMatrix) where {T1<:CuMatrix,T2<:CuMatrix}
    return CuMatrix{promote_type(eltype(T1), eltype(T2))}(mat_out)
end
Base.@propagate_inbounds function _promote_output_matrix(::Type{T1}, ::Type{T2}, mat_out::AbstractMatrix) where {T1<:CuMatrix,T2<:AbstractMatrix}
    return CuMatrix{promote_type(eltype(T1), eltype(T2))}(mat_out)
end
Base.@propagate_inbounds function _promote_output_matrix(::Type{T1}, ::Type{T2}, mat_out::AbstractMatrix) where {T1<:AbstractMatrix,T2<:CuMatrix}
    return CuMatrix{promote_type(eltype(T1), eltype(T2))}(mat_out)
end
Base.@propagate_inbounds function _promote_output_matrix(::Type{T}, mat_out::AbstractMatrix, out_dim::Int) where {T<:CuMatrix}
    return CuMatrix{eltype(T)}(mat_out)
end
Base.@propagate_inbounds function _promote_output_matrix(::Type{T}, mat_out::AbstractMatrix, out_dim::Tuple) where {T<:CuMatrix}
    return CuMatrix{eltype(T)}(mat_out)
end

# Determinants.
#
# `LinearAlgebra.det`/`logdet` go through LU and then read the diagonal of the
# factor one element at a time, which CUDA disallows. Every matrix Gabs takes a
# determinant of is a covariance matrix, so it is symmetric positive definite and
# Cholesky applies -- and is both cheaper and more stable than LU here.

function _det(A::CuMatrix)
    d = diag(cholesky(Symmetric(A)).U)
    return prod(d)^2
end

function _logdet(A::CuMatrix)
    d = diag(cholesky(Symmetric(A)).U)
    return 2 * sum(log, d)
end

# The symplectic form has to be built on the device that holds the data it
# multiplies; `symplecticform(basis)` alone is always a host `Matrix{Float64}`.

_symplecticform(basis::SymplecticBasis, x::CuArray) =
    CuMatrix{real(eltype(x))}(symplecticform(basis))

_complexform(basis::SymplecticBasis, x::CuArray) =
    CuMatrix{complex(real(eltype(x)))}(symplecticform(basis))

# Auxiliary matrices (a generaldyne projection, a homodyne squeezing term) are
# assembled on the host; they have to reach the device before being combined
# with moments that live there.
_like(x::CuArray, A::AbstractMatrix) = CuMatrix{eltype(x)}(A)
_like(::CuArray, A::CuMatrix) = A

# A tensor product may pair a device operand with a host one; the result belongs
# on the device, so the host side is moved there before they are combined.
_codevice(A::CuArray, B::CuArray) = (A, B)
_codevice(A::CuArray, B::AbstractArray) = (A, CuArray{promote_type(eltype(A), eltype(B))}(B))
_codevice(A::AbstractArray, B::CuArray) = (CuArray{promote_type(eltype(A), eltype(B))}(A), B)
