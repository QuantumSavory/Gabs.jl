# promote_type(CuVector{T}, Vector{T}) is Any, so the mixed pairs need their own
# methods. The output argument is annotated to keep the two arities disjoint.

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

# det/logdet read the LU diagonal elementwise, which CUDA disallows. These
# matrices are covariances, so Cholesky applies.

function _det(A::CuMatrix)
    d = diag(cholesky(Symmetric(A)).U)
    return prod(d)^2
end

function _logdet(A::CuMatrix)
    d = diag(cholesky(Symmetric(A)).U)
    return 2 * sum(log, d)
end

_symplecticform(basis::SymplecticBasis, x::CuArray) =
    CuMatrix{real(eltype(x))}(symplecticform(basis))

_complexform(basis::SymplecticBasis, x::CuArray) =
    CuMatrix{complex(real(eltype(x)))}(symplecticform(basis))

_like(x::CuArray, A::AbstractMatrix) = CuMatrix{eltype(x)}(A)
_like(::CuArray, A::CuMatrix) = A

_codevice(A::CuArray, B::CuArray) = (A, B)
_codevice(A::CuArray, B::AbstractArray) = (A, CuArray{promote_type(eltype(A), eltype(B))}(B))
_codevice(A::AbstractArray, B::CuArray) = (CuArray{promote_type(eltype(A), eltype(B))}(A), B)
