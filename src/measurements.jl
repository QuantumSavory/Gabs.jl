abstract type AbstractGaussianMeasurement end

"""
    _part_state(state::GaussianState{<:QuadPairBasis,M,V}, indices::Vector) -> a, b, A, B, C
    _part_state(state::GaussianState{<:QuadPairBasis,M,V}, indices::Int) -> a, b, A, B, C

Low-level function that partitions `state` into subsystems A and B, 
the latter system's modes specified by `indices`. The vectors `a` and `b`
are mean vectors of systems A and B, respectively. The matrices `A` and `B`
are covariance matrices of systems A and B, respectively. Matrix `C` is the 
correlation matrix between A and B.
"""
function _part_state(state::GaussianState{<:QuadPairBasis,M,V}, indices::I) where {M,V,I}
	return _partition(state, indices)
end
function _part_state(state::GaussianState{<:QuadBlockBasis,M,V}, indices::I) where {M,V,I}
	return _partition(state, indices)
end

# The partition is a pair of gathers on the quadratures of the measured and
# unmeasured modes, so the two layouts differ only in `_quadindices`. Allocating
# by indexing the moments keeps the blocks in the state's own element type and
# on its own device, instead of forcing host `Float64`.
function _partition(state::GaussianState, indices)
	basis = state.basis
	notindices = setdiff(1:basis.nmodes, indices)
	mean, covar = state.mean, state.covar
	qb = _quadindices(basis, indices)
	qa = _quadindices(basis, notindices)
	a, b = mean[qa], mean[qb]
	A, B = covar[qa, qa], covar[qb, qb]
	C = covar[qa, qb]
	return a, b, A, B, C
end