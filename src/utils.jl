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

# hooks: det/logdet read the LU diagonal elementwise, which some array backends
# disallow; those define these over a factorization they support
_det(A) = det(A)
_logdet(A) = logdet(A)
