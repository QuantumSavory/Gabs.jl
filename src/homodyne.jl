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

# Keyword arguments
- `rng::AbstractRNG = Random.default_rng()`: Random number generator that determines a random projection.
- `squeeze::Real = 1e-12`: Finite squeezing parameter.

# Examples
```
julia> st = squeezedstate(QuadBlockBasis(3), 1.0, pi/4);

julia> M = homodyne(st, [1, 3], [0.0, pi/2])
Homodyne{Vector{Float64}, GaussianState{QuadBlockBasis{Int64}, Vector{Float64}, Matrix{Float64}}}
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
    angles::G;
    rng::AbstractRNG = Random.default_rng(),
    squeeze::Real = 1e-12
) where {Tm,Tc,R,G}
    return _homodyne(rng, state, indices, angles, squeeze, Tm, Tc)
end
function homodyne(
    state::GaussianState{<:QuadBlockBasis,Tm,Tc},
    indices::R,
    angles::G;
    rng::AbstractRNG = Random.default_rng(),
    squeeze::Real = 1e-12
) where {Tm,Tc,R,G}
    return _homodyne(rng, state, indices, angles, squeeze, Tm, Tc)
end

# As `_generaldyne`: the conditional moments of the unmeasured modes scattered
# back onto a vacuum background.
function _homodyne(rng::AbstractRNG, state::GaussianState, indices, angles, squeeze::Real, ::Type{Tm}, ::Type{Tc}) where {Tm,Tc}
    basis = state.basis
    nmodes = basis.nmodes
    indlength = length(indices)
    indlength <= nmodes || throw(ArgumentError(Gabs.INDEX_ERROR))
    indlength == length(angles) || throw(ArgumentError(Gabs.GENERALDYNE_ERROR))
    result′, a, A = _homodyne_filter(rng, state, indices, angles; squeeze)
    notindices = setdiff(1:nmodes, indices)
    q = _quadindices(basis, notindices)
    mean′ = similar(state.mean, 2*nmodes)
    fill!(mean′, zero(eltype(state.mean)))
    covar′ = similar(state.covar, 2*nmodes, 2*nmodes)
    fill!(covar′, zero(eltype(state.covar)))
    covar′[diagind(covar′)] .= eltype(state.covar)(state.ħ/2)
    mean′[q] = a
    covar′[q, q] = A
    mean′′ = Gabs._promote_output_vector(Tm, mean′, 2*nmodes)
    covar′′ = Gabs._promote_output_matrix(Tc, covar′, 2*nmodes)
    state′ = GaussianState(basis, mean′′, covar′′, ħ = state.ħ)
    return Homodyne(result′, state′)
end
homodyne(rng::AbstractRNG, state::GaussianState{<:QuadPairBasis,Tm,Tc}, indices::R, angles::G; squeeze::Real = 1e-12) where {Tm,Tc,R,G} = homodyne(state, indices, angles; rng, squeeze)
homodyne(rng::AbstractRNG, state::GaussianState{<:QuadBlockBasis,Tm,Tc}, indices::R, angles::G; squeeze::Real = 1e-12) where {Tm,Tc,R,G} = homodyne(state, indices, angles; rng, squeeze)

"""
    rand([rng::AbstractRNG], ::Type{Homodyne}, state::GaussianState, indices::Vector, angles::Vector; shots = 1, squeeze = 1e-12) -> Array

Compute the projection of the subsystem of a Gaussian state `state` indicated by `indices`
on rotated quadrature states with homodyne phases given by `angles` and return an array of measured modes.
The number of shots is given by `shots`, which determines how many repeated and random homodyne measurements
are performed on the quantum system. The `squeeze` parameter determines the finite squeezing performed during the projection.

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
    shots::Int = 1,
    rng::AbstractRNG = Random.default_rng(),
    squeeze::Real = 1e-12
) where {Tm,Tc,R,G}
    return Base.rand(rng, Homodyne, state, indices, angles; shots, squeeze)
end
function Base.rand(
    rng::AbstractRNG,
    ::Type{Homodyne},
    state::GaussianState{<:QuadPairBasis,Tm,Tc},
    indices::R,
    angles::G;
    shots::Int = 1,
    squeeze::Real = 1e-12
) where {Tm,Tc,R,G}
    return _homodyne_samples(rng, state, indices, angles, shots, squeeze)
end
function Base.rand(
    ::Type{Homodyne}, 
    state::GaussianState{<:QuadBlockBasis,Tm,Tc}, 
    indices::R, 
    angles::G;
    shots::Int = 1,
    rng::AbstractRNG = Random.default_rng(),
    squeeze::Real = 1e-12
) where {Tm,Tc,R,G}
    return Base.rand(rng, Homodyne, state, indices, angles; shots, squeeze)
end
function Base.rand(
    rng::AbstractRNG,
    ::Type{Homodyne},
    state::GaussianState{<:QuadBlockBasis,Tm,Tc},
    indices::R,
    angles::G;
    shots::Int = 1,
    squeeze::Real = 1e-12
) where {Tm,Tc,R,G}
    return _homodyne_samples(rng, state, indices, angles, shots, squeeze)
end

# Homodyne outcomes are Gaussian with mean `b` and covariance `B` broadened by
# the finite-squeezing term; as in `_generaldyne_samples`, all shots come from
# one triangular product.
function _homodyne_samples(rng::AbstractRNG, state::GaussianState, indices, angles, shots::Int, squeeze::Real)
    basis = state.basis
    indlength = length(indices)
    indlength <= basis.nmodes || throw(ArgumentError(Gabs.INDEX_ERROR))
    indlength == length(angles) || throw(ArgumentError(Gabs.GENERALDYNE_ERROR))
    _, b, _, B, _ = _part_state(state, indices)
    B = B .+ _like(B, _squeezeaxes(basis, indlength, angles, squeeze))
    # `F.U'` names the same factor as `F.L`, but a backend whose Cholesky stores
    # the upper factor builds `.L` by transposing element by element, which is not
    # available on every array type.
    L = cholesky(Symmetric(B)).U'
    z = similar(b, 2*indlength, shots)
    randn!(rng, z)
    return L * z .+ b
end

function _homodyne_filter(
    rng::AbstractRNG,
    state::GaussianState{<:QuadPairBasis,Tm,Tc}, 
    indices::R, 
    angles::G;
    squeeze::Real = 1e-12
) where {Tm,Tc,R,G}
    basis = state.basis
    indlength = length(indices)
    nmodes′ = basis.nmodes - indlength
    a, b, A, B, C = _part_state(state, indices)
    B = B .+ _like(B, _squeezeaxes(basis, indlength, angles, squeeze))
    # sample from probability distribution by taking the displaced 
    # Cholesky decomposition of the covariance matrix
    symB = Symmetric(B)
    L = cholesky(symB).U'
    z = similar(b, 2*indlength)
    randn!(rng, z)
    resultmean = L * z .+ b
    meandiff = resultmean .- b
    # conditional mapping (see Serafini's Quantum Continuous Variables textbook for reference)
    buf = C * inv(symB)
    a .+= buf * meandiff
    A .-= buf * C'
    # promote output array type to ensure it matches the input array type
    result′ = Gabs._promote_output_vector(Tm, resultmean, 2*indlength)
    return result′, a, A
end
function _homodyne_filter(
    rng::AbstractRNG,
    state::GaussianState{<:QuadBlockBasis,Tm,Tc}, 
    indices::R, 
    angles::G;
    squeeze = 1e-12
) where {Tm,Tc,R,G}
    basis = state.basis
    indlength = length(indices)
    nmodes′ = basis.nmodes - indlength
    a, b, A, B, C = _part_state(state, indices)
    B = B .+ _like(B, _squeezeaxes(basis, indlength, angles, squeeze))
    # sample from probability distribution by taking the displaced 
    # Cholesky decomposition of the covariance matrix
    symB = Symmetric(B)
    L = cholesky(symB).U'
    z = similar(b, 2*indlength)
    randn!(rng, z)
    resultmean = L * z .+ b
    meandiff = resultmean .- b
    # conditional mapping (see Serafini's Quantum Continuous Variables textbook for reference)
    buf = C * inv(symB)
    a .+= buf * meandiff
    A .-= buf * C'
    # promote output array type to ensure it matches the input array type
    result′ = Gabs._promote_output_vector(Tm, resultmean, 2*indlength)
    return result′, a, A
end

# Finite squeezing along the axes given by `angles`, added to the measured
# block's covariance. The 2x2 rotation lives on the quadratures of each measured
# mode, whose positions within that block `_blockquadpositions` supplies, so one
# routine covers both layouts. It is assembled on the host and added in one
# operation, which keeps the caller's array backend untouched.
function _squeezeaxes(basis::SymplecticBasis, indlength::Int, angles, squeeze::Real)
    T = float(eltype(angles))
    N = zeros(T, 2*indlength, 2*indlength)::Matrix{T}
    # positions of mode i's two quadratures within the measured block
    qpos, ppos = _blockquadpositions(basis, indlength)
    @inbounds for i in Base.OneTo(indlength)
        θ = angles[i]
        ct, st = cos(θ), sin(θ)
        qi, pi_ = qpos(i), ppos(i)
        N[qi, qi]  += ct^2 * squeeze + st^2 / squeeze
        N[qi, pi_] += ct * st * (squeeze - 1 / squeeze)
        N[pi_, qi] += ct * st * (squeeze - 1 / squeeze)
        N[pi_, pi_] += st^2 * squeeze + ct^2 / squeeze
    end
    return N
end

# The measured block is itself laid out in the state's basis, so mode i owns
# rows (2i-1, 2i) pairwise and (i, i+indlength) blockwise.
_blockquadpositions(::QuadPairBasis, l::Int) = (i -> 2i - 1, i -> 2i)
_blockquadpositions(::QuadBlockBasis, l::Int) = (i -> i, i -> i + l)
