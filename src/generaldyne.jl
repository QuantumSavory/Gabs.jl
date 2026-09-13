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
					 proj::S = Matrix{eltype(Tc)}((state.ħ/2)*I, 2*length(indices), 2*length(indices))) where {Tm,Tc,R,S<:Union{Matrix,GaussianState}}
	basis = state.basis
	nmodes = basis.nmodes
	indlength = length(indices)
	indlength <= nmodes || throw(ArgumentError(INDEX_ERROR))
	if proj isa Matrix
		2*indlength == size(proj)[1] == size(proj)[2] || throw(ArgumentError(GENERALDYNE_ERROR))
	elseif proj isa GaussianState
		2*indlength == length(proj.mean) || throw(ArgumentError(GENERALDYNE_ERROR))
	end
	result′, a, A = _generaldyne_filter(state, indices, proj)
	mean′, covar′ = zeros(eltype(Tm), 2*nmodes), Matrix{eltype(Tc)}((state.ħ/2)*I, 2*nmodes, 2*nmodes)
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
	mean′′ = _promote_output_vector(Tm, mean′, 2*nmodes)
    covar′′ = _promote_output_matrix(Tc, covar′, 2*nmodes)
    state′ = GaussianState(basis, mean′′, covar′′, ħ = state.ħ)
	return Generaldyne(result′, state′)
end
function generaldyne(state::GaussianState{<:QuadBlockBasis,Tm,Tc}, indices::R; 
	proj::S = Matrix{eltype(Tc)}((state.ħ/2)*I, 2*length(indices), 2*length(indices))) where {Tm,Tc,R,S<:Union{Matrix,GaussianState}}
	basis = state.basis
	nmodes = basis.nmodes
	indlength = length(indices)
	indlength <= nmodes || throw(ArgumentError(INDEX_ERROR))
	if proj isa Matrix
		2*indlength == size(proj)[1] == size(proj)[2] || throw(ArgumentError(GENERALDYNE_ERROR))
	elseif proj isa GaussianState
		2*indlength == length(proj.mean) || throw(ArgumentError(GENERALDYNE_ERROR))
	end
	result′, a, A = _generaldyne_filter(state, indices, proj)
	mean′, covar′ = zeros(eltype(Tm), 2*nmodes), Matrix{eltype(Tc)}((state.ħ/2)*I, 2*nmodes, 2*nmodes)
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
				   shots::Int = 1, proj::S = Matrix{eltype(Tc)}((state.ħ/2)*I, 2*length(indices), 2*length(indices))) where {Tm,Tc,R,S<:Matrix}
	basis = state.basis
	indlength = length(indices)
	nmodes′ = basis.nmodes - indlength
	indlength <= basis.nmodes || throw(ArgumentError(INDEX_ERROR))
	2*indlength == size(proj)[1] == size(proj)[2] || throw(ArgumentError(GENERALDYNE_ERROR))
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
	# generate random mean vector samples
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
function Base.rand(::Type{Generaldyne}, state::GaussianState{<:QuadBlockBasis,Tm,Tc}, indices::R;
				   shots::Int = 1, proj::S = Matrix{eltype(Tc)}((state.ħ/2)*I, 2*length(indices), 2*length(indices))) where {Tm,Tc,R,S<:Matrix}
	basis = state.basis
	nmodes = basis.nmodes
	indlength = length(indices)
	nmodes′ = nmodes - indlength
	indlength <= nmodes || throw(ArgumentError(INDEX_ERROR))
	2*indlength == size(proj)[1] == size(proj)[2] || throw(ArgumentError(GENERALDYNE_ERROR))
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
	# generate random mean vector samples
	B .+= proj
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

function _generaldyne_filter(state::GaussianState{<:SymplecticBasis,Tm,Tc}, indices::R, proj::S) where {Tm,Tc,R,S<:Matrix}
	basis = state.basis
	indlength = length(indices)
	nmodes′ = basis.nmodes - indlength
	a, b, A, B, C = _part_state(state, indices)
	# generate random mean vector samples
	B .+= proj
	symB = Symmetric(B)
	L = cholesky(symB).L
	resultmean = L * randn(2*indlength) + b
	meandiff = resultmean - b
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
	B .+= proj.covar
	symB = Symmetric(B)
	meandiff = proj.mean - b
	# conditional mapping (see Serafini's Quantum Continuous Variables textbook for reference)
	buf = C * inv(symB)
	a .+= buf * meandiff
	A .-= buf * C'
	result′ = proj
	return result′, a, A
end
