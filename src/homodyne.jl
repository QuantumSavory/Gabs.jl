struct Homodyne{R,S<:GaussianState} <: Gabs.AbstractGaussianMeasurement
    result::R
    state::S
    function Homodyne(r::R, s::S) where {R,S<:GaussianState}
        return new{R,S}(r, s)
    end
end

# iteration for destructuring into components
Base.iterate(F::Homodyne) = (F.result, Val(:state))
Base.iterate(F::Homodyne, ::Val{:state}) = (F.state, Val(:done))
Base.iterate(F::Homodyne, ::Val{:done}) = nothing

# printing method
function Base.show(
    io::IO, 
    mime::MIME{Symbol("text/plain")}, 
    H::Homodyne{<:Any,<:GaussianState}
)
    Base.summary(io, H); println(io)
    println(io, "result:")
    Base.show(io, mime, H.result)
    println(io, "\noutput state:")
    Base.show(io, mime, H.state)
end

"""
    homodyne(state::GaussianState, indices::Vector, angles::Vector) -> Homodyne
    homodyne(state::GaussianState, index::Int, angle::Float64) -> Homodyne

Compute the projection of the subsystem of a Gaussian state `state` indicated by `indices`
on rotated quadrature states with homodyne phases given by `angles` and return a `Homodyne` object. 
The `result` and mapped state `output` can be obtained from the Homodyne object `M` via `M.result` and `M.output`.
Iterating the decomposition produces the components `result` and `output`.

Note the measured modes are replaced with vacuum states after the homodyne measurement.

# Examples
```
julia> st = squeezedstate(QuadBlockBasis(3), 1.0, pi/4);

julia> M = homodyne(st, [1, 3], [0.0, pi/2])
Generaldyne{Vector{Float64}, GaussianState{QuadBlockBasis{Int64}, Vector{Float64}, Matrix{Float64}}}
result:
4-element Vector{Float64}:
        -0.10967467526473408
    431152.803479904
    199553.173299574
        2.6446626022416964
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
    0.0   1.19762  0.0  0.0  -2.56458  0.0
    0.0   0.0      1.0  0.0   0.0      0.0
    0.0   0.0      0.0  1.0   0.0      0.0
    0.0  -2.56458  0.0  0.0   6.32677  0.0
    0.0   0.0      0.0  0.0   0.0      1.0

julia> result, state = M; # destructuring via iteration

julia> result == M.result && state == M.state
true
```
"""
function homodyne(
    state::GaussianState{<:QuadPairBasis,Tm,Tc}, 
    indices::R, 
    angles::G
) where {Tm,Tc,R,G}
    basis = state.basis
    nmodes = basis.nmodes
    indlength = length(indices)
    indlength < nmodes || throw(ArgumentError(Gabs.INDEX_ERROR))
    indlength == length(angles) || throw(ArgumentError(Gabs.GENERALDYNE_ERROR))
    # perform conditional mapping of Gaussian quantum state
    result′, a, A = _homodyne_filter(state, indices, angles)
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
    return Homodyne(result′, state′)
end
function homodyne(
    state::GaussianState{<:QuadBlockBasis,Tm,Tc}, 
    indices::R, 
    angles::G
) where {Tm,Tc,R,G}
    basis = state.basis
    nmodes = basis.nmodes
    indlength = length(indices)
    indlength < nmodes || throw(ArgumentError(Gabs.INDEX_ERROR))
    indlength == length(angles) || throw(ArgumentError(Gabs.GENERALDYNE_ERROR))
    # perform conditional mapping of Gaussian quantum state
    result′, a, A = _homodyne_filter(state, indices, angles)
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
    return Homodyne(result′, state′)
end

"""
    rand(::Type{Homodyne}, state::GaussianState, indices::Vector, angles::Vector; shots = 1) -> Array

Compute the projection of the subsystem of a Gaussian state `state` indicated by `indices`
on rotated quadrature states with homodyne phases given by `angles` and return an array of measured modes.
The number of shots is given by `shots`, which determines how many repeated and random homodyne measurements
are performed on the quantum system.

The output is an `2*length(indices) × shots` array, which contains the measured position and momentum modes columnwise
for each measurement, the ordering basis of the input Gaussian state `state`.

# Examples
```
julia > st = squeezedstate(QuadBlockBasis(3), 1.0, pi/4);

julia> rand(Homodyne, st, [1, 3], [pi/2, 0], shots = 5)
4×5 Matrix{Float64}:
8.53668e5   5.23331e5  -8.46171e5   4.66993e5  -1.093e6
-1.8943     -0.388814    0.179409    0.245702   -0.896928
-1.77362    -3.96152     0.351279   -3.2279     -1.74368
1.09432e6  -7.7091e5   -2.0881e5    1.31099e6  -5.16098e5
```
"""
function Base.rand(
    ::Type{Homodyne}, 
    state::GaussianState{<:QuadPairBasis,Tm,Tc}, 
    indices::R, 
    angles::G; 
    shots::Int = 1
) where {Tm,Tc,R,G}
    basis = state.basis
    indlength = length(indices)
    indlength < basis.nmodes || throw(ArgumentError(Gabs.INDEX_ERROR))
    indlength == length(angles) || throw(ArgumentError(Gabs.GENERALDYNE_ERROR))
    nmodes′ = basis.nmodes - indlength
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
    # infinite squeezing along axis defined by `angles`
    @inbounds for i in Base.OneTo(indlength)
        θ = angles[i]
        sq = 1e-12
        ct, st = cos(θ), sin(θ)
        B[i,i] += ct^2 * sq + st^2 / sq
        B[i,i+indlength] += ct * st * (sq - 1 / sq)
        B[i+indlength,i] += ct * st * (sq - 1 / sq)
        B[i+indlength,i+indlength] += st^2 * sq + ct^2 / sq
    end
    # sample from probability distribution by taking the displaced 
    # Cholesky decomposition of the covariance matrix
    symB = Symmetric(B)
    L = cholesky(symB).L
    buf = zeros(2*indlength)
    results = zeros(2*indlength, shots)
    @inbounds for i in Base.OneTo(shots)
        mul!(@view(results[:,i]), L, randn!(buf))
        @view(results[:,i]) .+= b
    end
    return results
end
function Base.rand(
    ::Type{Homodyne}, 
    state::GaussianState{<:QuadBlockBasis,Tm,Tc}, 
    indices::R, 
    angles::G;
    shots::Int = 1
) where {Tm,Tc,R,G}
    basis = state.basis
    nmodes = basis.nmodes
    indlength = length(indices)
    indlength < nmodes || throw(ArgumentError(Gabs.INDEX_ERROR))
    indlength == length(angles) || throw(ArgumentError(Gabs.GENERALDYNE_ERROR))
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
    # infinite squeezing along axis defined by `angles`
    @inbounds for i in Base.OneTo(indlength)
        θ = angles[i]
        sq = 1e-12
        ct, st = cos(θ), sin(θ)
        B[i,i] += ct^2 * sq + st^2 / sq
        B[i,i+indlength] += ct * st * (sq - 1 / sq)
        B[i+indlength,i] += ct * st * (sq - 1 / sq)
        B[i+indlength,i+indlength] += st^2 * sq + ct^2 / sq
    end
    # sample from probability distribution by taking the displaced 
    # Cholesky decomposition of the covariance matrix
    symB = Symmetric(B)
    L = cholesky(symB).L
    buf = zeros(2*indlength)
    results = zeros(2*indlength, shots)
    @inbounds for i in Base.OneTo(shots)
        mul!(@view(results[:,i]), L, randn!(buf))
        @view(results[:,i]) .+= b
    end
    return results
end

function _homodyne_filter(
    state::GaussianState{<:QuadPairBasis,Tm,Tc}, 
    indices::R, 
    angles::G
) where {Tm,Tc,R,G}
    basis = state.basis
    indlength = length(indices)
    nmodes′ = basis.nmodes - indlength
    a, b, A, B, C = _part_state(state, indices)
    # infinite squeezing along axis defined by `angles`
    @inbounds for i in Base.OneTo(indlength)
        θ = angles[i]
        sq = 1e-12
        ct, st = cos(θ), sin(θ)
        B[2i-1,2i-1] += ct^2 * sq + st^2 / sq
        B[2i-1,2i] += ct * st * (sq - 1 / sq)
        B[2i,2i-1] += ct * st * (sq - 1 / sq)
        B[2i,2i] += st^2 * sq + ct^2 / sq
    end
    # sample from probability distribution by taking the displaced 
    # Cholesky decomposition of the covariance matrix
    symB = Symmetric(B)
    L = cholesky(symB).L
    resultmean = L * randn(2*indlength) + b
    meandiff = resultmean - b
    # conditional mapping (see Serafini's Quantum Continuous Variables textbook for reference)
    buf = C * inv(symB)
    a .+= buf * meandiff
    A .-= buf * C'
    # promote output array type to ensure it matches the input array type
    result′ = Gabs._promote_output_vector(Tm, resultmean, 2*indlength)
    return result′, a, A
end
function _homodyne_filter(
    state::GaussianState{<:QuadBlockBasis,Tm,Tc}, 
    indices::R, 
    angles::G
) where {Tm,Tc,R,G}
    basis = state.basis
    indlength = length(indices)
    nmodes′ = basis.nmodes - indlength
    a, b, A, B, C = _part_state(state, indices)
    # infinite squeezing along axis defined by `angles`
    @inbounds for i in Base.OneTo(indlength)
        θ = angles[i]
        sq = 1e-12
        ct, st = cos(θ), sin(θ)
        B[i,i] += ct^2 * sq + st^2 / sq
        B[i,i+indlength] += ct * st * (sq - 1 / sq)
        B[i+indlength,i] += ct * st * (sq - 1 / sq)
        B[i+indlength,i+indlength] += st^2 * sq + ct^2 / sq
    end
    # sample from probability distribution by taking the displaced 
    # Cholesky decomposition of the covariance matrix
    symB = Symmetric(B)
    L = cholesky(symB).L
    resultmean = L * randn(2*indlength) + b
    meandiff = resultmean - b
    # conditional mapping (see Serafini's Quantum Continuous Variables textbook for reference)
    buf = C * inv(symB)
    a .+= buf * meandiff
    A .-= buf * C'
    # promote output array type to ensure it matches the input array type
    result′ = Gabs._promote_output_vector(Tm, resultmean, 2*indlength)
    return result′, a, A
end