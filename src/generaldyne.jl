struct Generaldyne{R,S<:GaussianState} <: AbstractGaussianMeasurement
	result::R
	state::S
	function Generaldyne(r::R, s::S) where {R,S<:GaussianState}
		return new{R,S}(r, s)
	end
end

# iteration for destructuring into components
Base.iterate(F::Generaldyne) = (F.result, Val(:state))
Base.iterate(F::Generaldyne, ::Val{:state}) = (F.state, Val(:done))
Base.iterate(F::Generaldyne, ::Val{:done}) = nothing

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, H::Generaldyne{<:Any,<:GaussianState})
    Base.summary(io, H); println(io)
    println(io, "result:")
    Base.show(io, mime, H.result)
    println(io, "\noutput state:")
    Base.show(io, mime, H.state)
end

"""
    generaldyne(state::GaussianState, indices::Vector; proj = (ħ/2)I) -> Generaldyne
    generaldyne(state::GaussianState, index::Int; proj = (ħ/2)I) -> Generaldyne

Compute the projection of the subsystem of a Gaussian state `state` indicated by `indices`
on `proj` and return a `Generaldyne` object. The keyword argument `proj` can take the following forms:

- If `proj` is a matrix, then the subsystem is projected onto a Gaussian state with a randomly sampled mean and covariance matrix `result`.
- If `proj` is a Gaussian state, then the subsystem is projected onto `proj`.

The `result` and mapped state `output` can be obtained from the Generaldyne object `M` via `M.result` and `M.output`.
Iterating the decomposition produces the components `result` and `output`.

Note the measured modes are replaced with vacuum states after the general-dyne measurement.

# Examples
```
julia> st = squeezedstate(QuadBlockBasis(3), 1.0, pi/4);

julia> M = generaldyne(st, [1, 3])
Generaldyne{GaussianState{QuadBlockBasis{Int64}, Vector{Float64}, Matrix{Float64}}, GaussianState{QuadBlockBasis{Int64}, Vector{Float64}, Matrix{Float64}}}
result:
GaussianState for 2 modes.
  symplectic basis: QuadBlockBasis
mean: 4-element Vector{Float64}:
  0.26967410461090285
  1.4683993027500133
 -1.84631450059537
  0.16832788926417352
covariance: 4×4 Matrix{Float64}:
 1.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0
 0.0  0.0  1.0  0.0
 0.0  0.0  0.0  1.0
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
function generaldyne(state::GaussianState{<:QuadPairBasis,Tm,Tc}, indices::R;
					 proj::S = Matrix{eltype(Tc)}((state.ħ/2)*I, 2*length(indices), 2*length(indices))) where {Tm,Tc,R,S<:Union{AbstractMatrix,GaussianState}}
	return _generaldyne(state, indices, proj, Tm, Tc)
end
function generaldyne(state::GaussianState{<:QuadBlockBasis,Tm,Tc}, indices::R;
	proj::S = Matrix{eltype(Tc)}((state.ħ/2)*I, 2*length(indices), 2*length(indices))) where {Tm,Tc,R,S<:Union{AbstractMatrix,GaussianState}}
	return _generaldyne(state, indices, proj, Tm, Tc)
end

# The conditional state of the unmeasured modes, written back into a vacuum
# background at the quadratures those modes own. This is the inverse of the
# gather in `_part_state`, so the two layouts again differ only in
# `_quadindices`.
function _generaldyne(state::GaussianState, indices, proj, ::Type{Tm}, ::Type{Tc}) where {Tm,Tc}
	basis = state.basis
	nmodes = basis.nmodes
	indlength = length(indices)
	indlength <= nmodes || throw(ArgumentError(INDEX_ERROR))
	if proj isa AbstractMatrix
		2*indlength == size(proj, 1) == size(proj, 2) || throw(ArgumentError(GENERALDYNE_ERROR))
	elseif proj isa GaussianState
		2*indlength == length(proj.mean) || throw(ArgumentError(GENERALDYNE_ERROR))
	end
	result′, a, A = _generaldyne_filter(state, indices, proj)
	notindices = setdiff(1:nmodes, indices)
	q = _quadindices(basis, notindices)
	mean′ = similar(state.mean, 2*nmodes)
	fill!(mean′, zero(eltype(state.mean)))
	covar′ = similar(state.covar, 2*nmodes, 2*nmodes)
	fill!(covar′, zero(eltype(state.covar)))
	covar′[diagind(covar′)] .= eltype(state.covar)(state.ħ/2)
	mean′[q] = a
	covar′[q, q] = A
	mean′′ = _promote_output_vector(Tm, mean′, 2*nmodes)
	covar′′ = _promote_output_matrix(Tc, covar′, 2*nmodes)
	state′ = GaussianState(basis, mean′′, covar′′, ħ = state.ħ)
	return Generaldyne(result′, state′)
end

"""
	rand(::Type{Generaldyne}, state::GaussianState, indices::Vector; shots = 1, proj = (ħ/2)I)

# Examples
```
julia > st = squeezedstate(QuadBlockBasis(3), 1.0, pi/4);

julia> rand(Generaldyne, st, [1, 3], shots = 5)
4×5 Matrix{Float64}:
  0.760996   1.24663    1.785     2.89803  -0.873372
  2.06074   -0.185524  -2.90446  -1.21932  -2.67317
  0.979994   2.44556   -2.20969  -4.12306  -1.31005
 -0.235823  -2.22807    1.11322   1.72146   1.37089
```
"""
function Base.rand(::Type{Generaldyne}, state::GaussianState{<:QuadPairBasis,Tm,Tc}, indices::R;
				   shots::Int = 1, proj::S = Matrix{eltype(Tc)}((state.ħ/2)*I, 2*length(indices), 2*length(indices))) where {Tm,Tc,R,S<:AbstractMatrix}
	return _generaldyne_samples(state, indices, proj, shots)
end
function Base.rand(::Type{Generaldyne}, state::GaussianState{<:QuadBlockBasis,Tm,Tc}, indices::R;
				   shots::Int = 1, proj::S = Matrix{eltype(Tc)}((state.ħ/2)*I, 2*length(indices), 2*length(indices))) where {Tm,Tc,R,S<:AbstractMatrix}
	return _generaldyne_samples(state, indices, proj, shots)
end

# Outcomes of the measured block are Gaussian with mean `b` and covariance
# `B + proj`; drawing them is one Cholesky factor applied to a matrix of normal
# deviates. Sampling every shot in a single product keeps the work in one
# `mul!`, and allocating the deviates with `similar` keeps them in the state's
# element type and on its device.
function _generaldyne_samples(state::GaussianState, indices, proj, shots::Int)
	basis = state.basis
	indlength = length(indices)
	indlength <= basis.nmodes || throw(ArgumentError(INDEX_ERROR))
	2*indlength == size(proj, 1) == size(proj, 2) || throw(ArgumentError(GENERALDYNE_ERROR))
	_, b, _, B, _ = _part_state(state, indices)
	B = B .+ _like(B, proj)
	# `F.U'` names the same factor as `F.L`, but a backend whose Cholesky stores
	# the upper factor builds `.L` by transposing element by element, which is not
	# available on every array type.
	L = cholesky(Symmetric(B)).U'
	z = similar(b, 2*indlength, shots)
	randn!(z)
	return L * z .+ b
end

function _generaldyne_filter(state::GaussianState{<:SymplecticBasis,Tm,Tc}, indices::R, proj::S) where {Tm,Tc,R,S<:AbstractMatrix}
	basis = state.basis
	indlength = length(indices)
	nmodes′ = basis.nmodes - indlength
	a, b, A, B, C = _part_state(state, indices)
	# generate random mean vector samples
	B = B .+ _like(B, proj)
	symB = Symmetric(B)
	L = cholesky(symB).U'
	z = similar(b, 2*indlength)
	randn!(z)
	resultmean = L * z .+ b
	meandiff = resultmean .- b
	# conditional mapping (see Serafini's Quantum Continuous Variables textbook for reference)
	buf = C * inv(symB)
	a .+= buf * meandiff
	A .-= buf * C'
	resultmean′ = _promote_output_vector(Tm, resultmean, 2*indlength)
	result′ = GaussianState(typeof(basis)(indlength), resultmean′, proj, ħ = state.ħ)
	return result′, a, A
end
function _generaldyne_filter(state::GaussianState{<:SymplecticBasis,Tm,Tc}, indices::R, proj::S) where {Tm,Tc,R,S<:GaussianState}
	basis = state.basis
	indlength = length(indices)
	nmodes′ = basis.nmodes - indlength
	a, b, A, B, C = _part_state(state, indices)
	B = B .+ _like(B, proj.covar)
	symB = Symmetric(B)
	meandiff = proj.mean .- b
	# conditional mapping (see Serafini's Quantum Continuous Variables textbook for reference)
	buf = C * inv(symB)
	a .+= buf * meandiff
	A .-= buf * C'
	result′ = proj
	return result′, a, A
end
