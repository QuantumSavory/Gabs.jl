##
# Matrix product state (MPS) representations of Gaussian states
##

"""
    mpsstate(state::GaussianState; cutoff::Int, maxdim::Int = typemax(Int), svdcutoff::Real = 1e-12)

Construct a matrix product state (MPS) representation, in the truncated Fock basis, of a
**pure** Gaussian `state`.

This function requires the `ITensors.jl` and `ITensorMPS.jl` package extension: run
`using ITensors, ITensorMPS` in addition to `using Gabs` to enable it. The returned object
is an `ITensorMPS.MPS` built on `Qudit`-type site indices of local dimension `cutoff`.

## Arguments

- `cutoff`: the local Fock-space (photon-number) truncation per mode, i.e. each site keeps
  Fock states `|0⟩, …, |cutoff-1⟩`. Larger `cutoff` reduces Fock-truncation error at the
  cost of a larger local tensor dimension.
- `maxdim`: the maximum MPS bond dimension kept after each gate application (SVD
  truncation). Defaults to unbounded.
- `svdcutoff`: the discarded-weight tolerance for SVD-based bond-dimension truncation after
  each gate application. Defaults to `1e-12`.

## Mathematical description

The construction follows Nüßeler, Dhand, Huelga, and Plenio, *Phys. Rev. A* **104**, 012415
(2021): the state's covariance matrix is Williamson-decomposed via [`williamson`](@ref) into
a symplectic matrix `S` (since `state` is pure, `covar ≈ (ħ/2) S Sᵀ`), which is then
Bloch-Messiah-decomposed via [`blochmessiah`](@ref) into passive-active-passive layers
`S = O Λ Q`. Each layer is implemented as a circuit of one- and two-mode Fock-basis gates
(single-mode phase shifts and squeezers, and a triangular mesh of beam splitters
implementing the passive `O`/`Q` transformations, following the Reck et al. decomposition),
applied in sequence to the vacuum product-state MPS with SVD truncation after each gate,
matching the MPO/MPS gate-application methodology also used by Yanagimoto, Ng, Wright,
Onodera, and Mabuchi, *Optica* **8**, 1306 (2021). A final layer of single-mode displacement
gates accounts for the state's mean vector.

## Scope (Phase 1)

Only **pure** Gaussian states are currently supported (checked via [`purity`](@ref)); mixed
states throw an error, since representing them as an MPS requires purification, which is
not yet implemented. Non-Gaussian states (e.g. [`GaussianLinearCombination`](@ref)) are also
not yet supported.

**Known limitation**: `SymplecticMatrices.jl`'s `blochmessiah` has been observed to
occasionally return a passive transformation that is not itself symplectic for certain
multi-mode states reached via [`williamson`](@ref) (entangled states such as
[`eprstate`](@ref) reproduce this reliably; independent per-mode products of single-mode
states do not). `mpsstate` detects this and raises an informative error rather than
silently returning an incorrect state — this is a property of the numerical decomposition,
not of the input state, so retrying will not help. Single-mode states are unaffected.

## Example

```julia
using Gabs, ITensors, ITensorMPS

basis = QuadPairBasis(1)
state = squeezedstate(basis, 0.5, 0.0) ⊗ squeezedstate(basis, 0.3, 0.7)
psi = mpsstate(state; cutoff = 20)
inner(psi, psi) # ≈ 1
```
"""
function mpsstate end

function mpsstate(state::GaussianState; kwargs...)
    throw(ErrorException(MPS_EXT_ERROR))
end
