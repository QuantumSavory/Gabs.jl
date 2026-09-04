##
# Reck et al. triangular-mesh decomposition of a passive N-mode unitary into a circuit of
# two-mode "BS" gates and a final layer of single-mode "Phase" gates (see `siteops.jl`).
#
# Convention: applying the returned gates, in order, to a ket via `mode-amplitude vector ↦
# gate matrix * amplitude vector` (i.e. exactly how `ITensorMPS.apply` composes gates onto
# an MPS) implements multiplication of the single-excitation-subspace amplitudes by `U`
# itself (not its inverse/conjugate/transpose). Validated against random Haar unitaries,
# both at the matrix level and by explicit ITensors MPS gate application, before use here.
##

"""
    _givens_params(z1::Complex, z2::Complex; atol = 1e-12) -> (θ, φ)

Given two complex numbers `z1, z2` (entries of a unitary matrix row), return the real
`(θ, φ)` such that `z1*cos(θ) - im*sin(θ)*cis(φ)*z2 = 0`, i.e. the parameters of the
two-mode gate (see `_embed_givens`) that zeroes `z1` when `z2` is its neighbor.
"""
function _givens_params(z1::Number, z2::Number; atol::Real = 1e-12)
    if abs(z1) < atol && abs(z2) < atol
        return 0.0, 0.0
    end
    θ = atan(abs(z1), abs(z2))
    φ = angle(-im * z1 * conj(z2))
    return θ, φ
end

"""
    _embed_givens(N::Int, c::Int, θ::Real, φ::Real) -> Matrix{ComplexF64}

`N×N` identity except for the 2×2 block on modes `(c, c+1)`:
`[[cos θ, -im sin θ cis(-φ)], [-im sin θ cis(φ), cos θ]]`, matching the physical two-mode
`"BS"` gate defined in `siteops.jl`.
"""
function _embed_givens(N::Int, c::Int, θ::Real, φ::Real)
    T = Matrix{ComplexF64}(I, N, N)
    T[c, c] = cos(θ)
    T[c, c+1] = -im * sin(θ) * cis(-φ)
    T[c+1, c] = -im * sin(θ) * cis(φ)
    T[c+1, c+1] = cos(θ)
    return T
end

"""
    _reck_decompose(U::AbstractMatrix{<:Complex}; atol = 1e-12) -> (gates, phases)

Decompose an `N×N` unitary `U` into a Reck et al. triangular mesh of two-mode gates
`gates::Vector{Tuple{Int,Float64,Float64}}` (each `(c, θ, φ)` acting on modes `(c, c+1)`)
followed by a final single-mode phase layer `phases::Vector{Float64}` (length `N`), such
that applying, **in order**, the two-mode `"BS"` gates and then the `"Phase"` gates to a
ket implements exactly `U` on the single-excitation (and, by the algebra of passive
transformations, every-excitation) subspace.
"""
function _reck_decompose(U::AbstractMatrix{<:Complex}; atol::Real = 1e-12)
    N = size(U, 1)
    size(U, 2) == N || throw(ArgumentError("U must be square"))
    Uwork = Matrix{ComplexF64}(U)
    gates = Tuple{Int,Float64,Float64}[]
    @inbounds for r in N:-1:2
        for c in 1:(r-1)
            z1, z2 = Uwork[r, c], Uwork[r, c+1]
            θ, φ = _givens_params(z1, z2; atol = atol)
            push!(gates, (c, θ, φ))
            Uwork = Uwork * _embed_givens(N, c, θ, φ)
        end
    end
    phases = angle.(diag(Uwork))
    circuit_gates = [(c, -θ, φ) for (c, θ, φ) in gates]
    return circuit_gates, phases
end

"""
    _passive_unitary(S::AbstractMatrix{<:Real}, basis::QuadPairBasis) -> Matrix{ComplexF64}

Extract the `N×N` complex unitary `U` (acting on annihilation operators) corresponding to
a real, `2N×2N` orthogonal-symplectic **passive** transformation `S` given in `basis`. Uses
[`Gabs.changebasis`](@ref) to convert to the grouped `QuadBlockBasis` form
`[[Re U, -Im U], [Im U, Re U]]`, from which `U` is read off directly.
"""
function _passive_unitary(S::AbstractMatrix{<:Real}, basis::QuadPairBasis)
    n = nmodes(basis)
    op = GaussianUnitary(basis, zeros(2n), Matrix(S))
    opblock = changebasis(QuadBlockBasis, op)
    M = opblock.symplectic
    ReU = @view M[1:n, 1:n]
    ImU = @view M[n+1:2n, 1:n]
    return ReU .+ im .* ImU
end

"""
    _regularize_symplectic(S::AbstractMatrix{<:Real}, basis::QuadPairBasis; epsilon = 1e-6)

`SymplecticMatrices.jl`'s `blochmessiah` has been observed to occasionally return an
`O, values, Q` decomposition where `O * Diagonal(...) * Q` reconstructs `S` poorly when
`S`'s singular values coincide exactly (most commonly when one or more modes are exactly
unsqueezed, i.e. a Bloch-Messiah value of exactly `1`). Since `mpsstate` calls `blochmessiah`
on a state built purely from `williamson`, this degeneracy is common (any mode not
entangled/squeezed relative to the others triggers it), so we perturb `S` by a tiny,
deterministic, per-mode symplectic squeeze before decomposing it, breaking exact
coincidences at a cost many orders of magnitude below the Fock-truncation and SVD-truncation
error already inherent to `mpsstate`. This does not fully eliminate `blochmessiah`
reliability issues (see the caller's `issymplectic` check on `O`/`Q`), but resolves the
specific degenerate-singular-value failure mode.
"""
function _regularize_symplectic(S::AbstractMatrix{<:Real}, basis::QuadPairBasis; epsilon::Real = 1e-6)
    n = nmodes(basis)
    r = [epsilon * i for i in 1:n]
    theta = zeros(n)
    P = squeeze(basis, r, theta).symplectic
    return Matrix(S) * P
end
