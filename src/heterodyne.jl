struct Heterodyne{R,S<:GaussianState} <: Gabs.AbstractGaussianMeasurement
    result::R
    state::S
    function Heterodyne(r::R, s::S) where {R,S<:GaussianState}
        return new{R,S}(r, s)
    end
end

# iteration for destructuring into components
Base.iterate(F::Heterodyne) = (F.result, Val(:state))
Base.iterate(F::Heterodyne, ::Val{:state}) = (F.state, Val(:done))
Base.iterate(F::Heterodyne, ::Val{:done}) = nothing

# printing method
function Base.show(
    io::IO, 
    mime::MIME{Symbol("text/plain")}, 
    H::Heterodyne{<:Any,<:GaussianState}
)
    Base.summary(io, H); println(io)
    println(io, "result:")
    Base.show(io, mime, H.result)
    println(io, "\noutput state:")
    Base.show(io, mime, H.state)
end

"""
    heterodyne(state::GaussianState, indices::Vector) -> Heterodyne
    heterodyne(state::GaussianState, index::Int) -> Heterodyne

Compute the heterodyne measurement of the subsystem of a Gaussian state `state` 
indicated by `indices` and return a `Heterodyne` object. 
The `result` (phase-space quadratures) and mapped state `output` can be obtained 
from the Heterodyne object `M` via `M.result` and `M.output`.
Iterating the decomposition produces the components `result` and `output`.

Heterodyne measurement simultaneously measures both quadratures of the mode(s) with unit gain.
The measured modes are replaced with vacuum states after the heterodyne measurement.

# Keyword arguments
- `rng::AbstractRNG = Random.default_rng()`: Random number generator that determines a random projection.

# Examples
```
julia> st = squeezedstate(QuadBlockBasis(3), 1.0, pi/4);

julia> M = heterodyne(st, [1, 3])
Heterodyne{Matrix{ComplexF64}, GaussianState{QuadBlockBasis{Int64}, Vector{Float64}, Matrix{Float64}}}
result:
2×2 Matrix{ComplexF64}:
  0.123456+0.234567im  0.345678+0.456789im
  0.567890+0.678901im  0.789012+0.890123im
output state:
GaussianState for 3 modes.
    symplectic basis: QuadBlockBasis
mean: 6-element Vector{Float64}:
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
covariance: 6×6 Matrix{Float64}:
    1.0   0.0      0.0  0.0   0.0      0.0
    0.0   1.0      0.0  0.0   0.0      0.0
    0.0   0.0      1.0  0.0   0.0      0.0
    0.0   0.0      0.0  1.0   0.0      0.0
    0.0   0.0      0.0  0.0   1.0      0.0
    0.0   0.0      0.0  0.0   0.0      1.0

julia> result, state = M; # destructuring via iteration

julia> result == M.result && state == M.state
true
```
"""
function heterodyne(
    state::GaussianState{<:QuadPairBasis,Tm,Tc}, 
    indices::R;
    rng::AbstractRNG = Random.default_rng()
) where {Tm,Tc,R}
    basis = state.basis
    nmodes = basis.nmodes
    indlength = length(indices)
    indlength < nmodes || throw(ArgumentError(Gabs.INDEX_ERROR))
    # perform conditional mapping of Gaussian quantum state
    result′, a, A = _heterodyne_filter(rng, state, indices)
    mean′ = zeros(eltype(Tm), 2*nmodes)
    covar′ = Matrix{eltype(Tc)}((state.ħ/2)*I, 2*nmodes, 2*nmodes)
    # fill in measured modes with vacuum states 
    notindices = setdiff(1:nmodes, indices)
    @inbounds for i in eachindex(notindices)
        idx = notindices[i]
        copyto!(@view(mean′[2idx-1:2idx]), @view(a[2i-1:2i]))
        @inbounds for j in i:length(notindices)
            otheridx = notindices[j]
            covar′[2*idx-1, 2*otheridx-1] = A[2*i-1, 2*j-1]
            covar′[2*idx-1, 2*otheridx] = A[2*i-1, 2*j]
            covar′[2*idx, 2*otheridx-1] = A[2*i, 2*j-1]
            covar′[2*idx, 2*otheridx] = A[2*i, 2*j]
            covar′[2*otheridx-1, 2*idx-1] = A[2*j-1, 2*i-1]
            covar′[2*otheridx-1, 2*idx] = A[2*j-1, 2*i]
            covar′[2*otheridx, 2*idx-1] = A[2*j, 2*i-1]
            covar′[2*otheridx, 2*idx] = A[2*j, 2*i]
        end
    end
    # promote output array type to ensure it matches the input array type
    mean′′ = Gabs._promote_output_vector(Tm, mean′, 2*nmodes)
    covar′′ = Gabs._promote_output_matrix(Tc, covar′, 2*nmodes)
    state′ = GaussianState(basis, mean′′, covar′′, ħ = state.ħ)
    return Heterodyne(result′, state′)
end
function heterodyne(
    state::GaussianState{<:QuadBlockBasis,Tm,Tc}, 
    indices::R;
    rng::AbstractRNG = Random.default_rng()
) where {Tm,Tc,R}
    basis = state.basis
    nmodes = basis.nmodes
    indlength = length(indices)
    indlength < nmodes || throw(ArgumentError(Gabs.INDEX_ERROR))
    # perform conditional mapping of Gaussian quantum state
    result′, a, A = _heterodyne_filter(rng, state, indices)
    mean′ = zeros(eltype(Tm), 2*nmodes)
    covar′ = Matrix{eltype(Tc)}((state.ħ/2)*I, 2*nmodes, 2*nmodes)
    nmodes′ = nmodes - length(indices)
    # fill in measured modes with vacuum states
    notindices = setdiff(1:nmodes, indices)
    @inbounds for i in eachindex(notindices)
        idx = notindices[i]
        mean′[idx] = a[i]
        mean′[idx+nmodes] = a[i+nmodes′]
        @inbounds for j in i:length(notindices)
            otheridx = notindices[j]
            covar′[idx,otheridx] = A[i,j]
            covar′[otheridx,idx] = A[j,i]
            covar′[idx+nmodes,otheridx] = A[i+nmodes′,j]
            covar′[idx,otheridx+nmodes] = A[i,j+nmodes′]
            covar′[otheridx,idx+nmodes] = A[j,i+nmodes′]
            covar′[otheridx+nmodes,idx] = A[j+nmodes′,i]
            covar′[idx+nmodes,otheridx+nmodes] = A[i+nmodes′,j+nmodes′]
            covar′[otheridx+nmodes,idx+nmodes] = A[j+nmodes′,i+nmodes′]
        end
    end
    # promote output array type to ensure it matches the input array type
    mean′′ = Gabs._promote_output_vector(Tm, mean′, 2*nmodes)
    covar′′ = Gabs._promote_output_matrix(Tc, covar′, 2*nmodes)
    state′ = GaussianState(basis, mean′′, covar′′, ħ = state.ħ)
    return Heterodyne(result′, state′)
end
heterodyne(rng::AbstractRNG, state::GaussianState{<:QuadPairBasis,Tm,Tc}, indices::R) where {Tm,Tc,R} = heterodyne(state, indices; rng)
heterodyne(rng::AbstractRNG, state::GaussianState{<:QuadBlockBasis,Tm,Tc}, indices::R) where {Tm,Tc,R} = heterodyne(state, indices; rng)

"""
    Base.rand(::Type{Heterodyne}, state::GaussianState, indices; shots=1, rng=Random.default_rng())

Generate random heterodyne measurement outcomes for a Gaussian state.
Returns a matrix with shape (nmodes_measured, shots) where each column is a complex outcome 
representing the two-quadrature measurement.
"""
function Base.rand(
    rng::AbstractRNG,
    ::Type{Heterodyne}, 
    state::GaussianState{<:QuadPairBasis,Tm,Tc}, 
    indices::R;
    shots::Int = 1
) where {Tm,Tc,R}
    basis = state.basis
    indlength = length(indices)
    indlength < basis.nmodes || throw(ArgumentError(Gabs.INDEX_ERROR))
    mean, covar = state.mean, state.covar
    # write mean and covariance matrix of measured modes to vector `b` and matrix `B`, respectively
    b, B = zeros(2*indlength), zeros(2*indlength, 2*indlength)
    @inbounds for i in eachindex(indices)
        idx = indices[i]
        b[2i-1:2i] .= @view(mean[2idx-1:2idx])
        @inbounds for j in eachindex(indices)
            otheridx = indices[j]
            if idx == otheridx
                B[2i-1:2i, 2i-1:2i] .= @view(covar[2idx-1:2idx, 2idx-1:2idx])
            else
                B[2i-1:2i, 2j-1:2j] .= @view(covar[2idx-1:2idx, 2otheridx-1:2otheridx])
                B[2j-1:2j, 2i-1:2i] .= @view(covar[2otheridx-1:2otheridx, 2idx-1:2idx])
            end
        end
    end
    # add vacuum noise (unit gain heterodyne measurement)
    @inbounds for i in Base.OneTo(indlength)
        B[2i-1, 2i-1] += 1.0
        B[2i, 2i] += 1.0
    end
    # sample from probability distribution by taking the displaced 
    # Cholesky decomposition of the covariance matrix
    symB = Symmetric(B)
    L = cholesky(symB).L
    buf = zeros(2*indlength)
    results = zeros(ComplexF64, indlength, shots)
    @inbounds for shot in Base.OneTo(shots)
        mul!(buf, L, randn!(rng, buf))
        buf .+= b
        @inbounds for i in Base.OneTo(indlength)
            results[i, shot] = complex(buf[2i-1], buf[2i])
        end
    end
    return results
end
function Base.rand(
    rng::AbstractRNG,
    ::Type{Heterodyne}, 
    state::GaussianState{<:QuadBlockBasis,Tm,Tc}, 
    indices::R;
    shots::Int = 1
) where {Tm,Tc,R}
    basis = state.basis
    nmodes = basis.nmodes
    indlength = length(indices)
    indlength < nmodes || throw(ArgumentError(Gabs.INDEX_ERROR))
    nmodes′ = nmodes - indlength
    mean, covar = state.mean, state.covar
    # write mean and covariance matrix of measured modes to vector `b` and matrix `B`, respectively
    b, B = zeros(2*indlength), zeros(2*indlength, 2*indlength)
    @inbounds for i in eachindex(indices)
        idx = indices[i]
        b[i] = mean[idx]
        b[i+indlength] = mean[idx+nmodes]
        @inbounds for j in eachindex(indices)
            otheridx = indices[j]
            if idx == otheridx
                B[i, i] = covar[idx, idx]
                B[i+indlength, i] = covar[idx+nmodes, idx]
                B[i, i+indlength] = covar[idx, idx+nmodes]
                B[i+indlength, i+indlength] = covar[idx+nmodes, idx+nmodes]
            else
                B[i, j] = covar[idx, otheridx]
                B[i+indlength, j] = covar[idx+nmodes, otheridx]
                B[i, j+indlength] = covar[idx, otheridx+nmodes]
                B[i+indlength, j+indlength] = covar[idx+nmodes, otheridx+nmodes]

                B[j, i] = covar[otheridx, idx]
                B[j+indlength, i] = covar[otheridx+nmodes, idx]
                B[j, i+indlength] = covar[otheridx, idx+nmodes]
                B[j+indlength, i+indlength] = covar[otheridx+nmodes, idx+nmodes]
            end
        end
    end
    # add vacuum noise (unit gain heterodyne measurement)
    @inbounds for i in Base.OneTo(indlength)
        B[i, i] += 1.0
        B[i+indlength, i+indlength] += 1.0
    end
    # sample from probability distribution by taking the displaced 
    # Cholesky decomposition of the covariance matrix
    symB = Symmetric(B)
    L = cholesky(symB).L
    buf = zeros(2*indlength)
    results = zeros(ComplexF64, indlength, shots)
    @inbounds for shot in Base.OneTo(shots)
        mul!(buf, L, randn!(rng, buf))
        buf .+= b
        @inbounds for i in Base.OneTo(indlength)
            results[i, shot] = complex(buf[i], buf[i+indlength])
        end
    end
    return results
end
function Base.rand(
    ::Type{Heterodyne}, 
    state::GaussianState, 
    indices::R;
    shots::Int = 1,
    rng::AbstractRNG = Random.default_rng()
) where {R}
    return Base.rand(rng, Heterodyne, state, indices; shots)
end

function _heterodyne_filter(
    rng::AbstractRNG,
    state::GaussianState{<:QuadPairBasis,Tm,Tc}, 
    indices::R
) where {Tm,Tc,R}
    basis = state.basis
    indlength = length(indices)
    nmodes′ = basis.nmodes - indlength
    a, b, A, B, C = _part_state(state, indices)
    # add unit-gain vacuum noise for heterodyne measurement
    @inbounds for i in Base.OneTo(indlength)
        B[2i-1,2i-1] += 1.0
        B[2i,2i] += 1.0
    end
    # sample from probability distribution by taking the displaced 
    # Cholesky decomposition of the covariance matrix
    symB = Symmetric(B)
    L = cholesky(symB).L
    resultmean = L * randn(rng, 2*indlength) + b
    meandiff = resultmean - b
    # conditional mapping
    buf = C * inv(symB)
    a .+= buf * meandiff
    A .-= buf * C'
    # convert to complex representation and promote output array type
    result_complex = zeros(ComplexF64, indlength)
    @inbounds for i in Base.OneTo(indlength)
        result_complex[i] = complex(resultmean[2i-1], resultmean[2i])
    end
    return result_complex, a, A
end
function _heterodyne_filter(
    rng::AbstractRNG,
    state::GaussianState{<:QuadBlockBasis,Tm,Tc}, 
    indices::R
) where {Tm,Tc,R}
    basis = state.basis
    indlength = length(indices)
    nmodes′ = basis.nmodes - indlength
    a, b, A, B, C = _part_state(state, indices)
    # add unit-gain vacuum noise for heterodyne measurement
    @inbounds for i in Base.OneTo(indlength)
        B[i,i] += 1.0
        B[i+indlength,i+indlength] += 1.0
    end
    # sample from probability distribution by taking the displaced 
    # Cholesky decomposition of the covariance matrix
    symB = Symmetric(B)
    L = cholesky(symB).L
    resultmean = L * randn(rng, 2*indlength) + b
    meandiff = resultmean - b
    # conditional mapping
    buf = C * inv(symB)
    a .+= buf * meandiff
    A .-= buf * C'
    # convert to complex representation
    result_complex = zeros(ComplexF64, indlength)
    @inbounds for i in Base.OneTo(indlength)
        result_complex[i] = complex(resultmean[i], resultmean[i+indlength])
    end
    return result_complex, a, A
end
