Base.@propagate_inbounds function _promote_output_vector(::Type{T1}, ::Type{T2}, vec_out) where {T1,T2}
    T = promote_type(T1, T2)
    T <: Vector{Float64} ? vec_out : T(vec_out)
end
Base.@propagate_inbounds function _promote_output_vector(::Type{T}, vec_out, vec_length::Tl) where {T,Tl<:Int}
    T <: Vector{Float64} ? vec_out : T(vec_out)
end
Base.@propagate_inbounds function _promote_output_matrix(::Type{T1}, ::Type{T2}, mat_out) where {T1,T2}
    T = promote_type(T1, T2)
    T <: Matrix{Float64} ? mat_out : T(mat_out)
end
Base.@propagate_inbounds function _promote_output_matrix(::Type{T}, mat_out, out_dim::Td) where {T,Td<:Int}
    T <: Matrix{Float64} ? mat_out : T(mat_out)
end
Base.@propagate_inbounds function _promote_output_matrix(::Type{T}, mat_out, out_dim::Td) where {T,Td<:Tuple}
    T <: Matrix{Float64} ? mat_out : T(mat_out)
end

"""
    _det(A)
    _logdet(A)

Determinant and log-determinant of a covariance-like matrix.

These wrap `det`/`logdet` so that array backends without an LU-based
implementation can supply their own. `LinearAlgebra.det` reads the diagonal of
the LU factorization element by element, which some GPU array types disallow;
such a backend defines these methods over a factorization it does support.
"""
_det(A) = det(A)
_logdet(A) = logdet(A)
