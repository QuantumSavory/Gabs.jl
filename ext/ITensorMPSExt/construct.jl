##
# Gaussian state -> MPS construction pipeline (see the docstring on `Gabs.mpsstate`).
##

function Gabs.mpsstate(
    state::GaussianState{<:QuadBlockBasis};
    kwargs...
)
    return Gabs.mpsstate(changebasis(QuadPairBasis, state); kwargs...)
end

function Gabs.mpsstate(
    state::GaussianState{<:QuadPairBasis};
    cutoff::Int,
    maxdim::Int = typemax(Int),
    svdcutoff::Real = 1e-12,
    purity_atol::Real = 1e-6,
)
    isapprox(purity(state), 1; atol = purity_atol) || throw(ErrorException(Gabs.MPS_MIXED_STATE_ERROR))

    basis = state.basis
    n = nmodes(basis)
    ħ = state.ħ

    S, _ = williamson(state)
    Sreg = _regularize_symplectic(S, basis)
    passiveop = GaussianUnitary(basis, zeros(2n), Sreg; ħ = ħ)
    O, values, Q = blochmessiah(passiveop)

    # `SymplecticMatrices.jl`'s `blochmessiah` has been observed to occasionally return an
    # `O`/`Q` that reconstructs `O * Diagonal(...) * Q ≈ Sreg` to good precision while `O`
    # (or `Q`) individually fails to be symplectic — i.e. an internally inconsistent
    # decomposition. Since everything downstream (`_passive_unitary`, the Reck mesh) assumes
    # a genuine orthogonal-symplectic `O`/`Q`, silently proceeding would in general produce a
    # wrong state, so this is checked for explicitly. The single-mode (`n == 1`) case is
    # exempt: there, `O`/`Q` reduce to single-site phase gates applied either to the vacuum
    # (always trivially exact, regardless of the phase) or immediately after the single
    # squeezing gate — empirically verified (see `test/test_mps.jl`) to always reconstruct
    # the correct state even when this defect is present.
    if n > 1 && (!issymplectic(basis, O; atol = 1e-6) || !issymplectic(basis, Q; atol = 1e-6))
        throw(ErrorException(Gabs.MPS_BLOCHMESSIAH_ERROR))
    end

    Uq = _passive_unitary(Q, basis)
    Uo = _passive_unitary(O, basis)
    gatesQ, phasesQ = _reck_decompose(Uq)
    gatesO, phasesO = _reck_decompose(Uo)
    rs = -log.(values)

    sites = siteinds("Qudit", n; dim = cutoff)
    psi = MPS(sites, "0")

    psi = _apply_mesh(psi, sites, gatesQ, phasesQ; maxdim = maxdim, svdcutoff = svdcutoff)
    for i in 1:n
        gate = op("Squeeze", sites[i]; r = rs[i], theta = 0.0)
        psi = apply(gate, psi)
    end
    psi = _apply_mesh(psi, sites, gatesO, phasesO; maxdim = maxdim, svdcutoff = svdcutoff)

    mean = state.mean
    for i in 1:n
        alpha = (mean[2i-1] + im * mean[2i]) / sqrt(2 * ħ)
        gate = op("Displace", sites[i]; alpha = alpha)
        psi = apply(gate, psi)
    end

    return psi
end

function _apply_mesh(psi::MPS, sites, gates, phases; maxdim::Int, svdcutoff::Real)
    for (c, θ, φ) in gates
        gate = op("BS", sites[c], sites[c+1]; theta = θ, phi = φ)
        psi = apply(gate, psi; maxdim = maxdim, cutoff = svdcutoff)
    end
    for (i, φ) in enumerate(phases)
        gate = op("Phase", sites[i]; theta = φ)
        psi = apply(gate, psi)
    end
    return psi
end
