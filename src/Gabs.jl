module Gabs

import LinearAlgebra
using LinearAlgebra: I, det, mul!, diag, qr, eigvals, Diagonal, cholesky, Symmetric, dot, Hermitian, logdet

import QuantumInterface: StateVector, AbstractOperator, apply!, tensor, ⊗, directsum, ⊕, entropy_vn, fidelity, logarithmic_negativity, ptrace, embed

import Random
using Random: randn!, AbstractRNG

import SymplecticMatrices: williamson, Williamson, polar, Polar, blochmessiah, BlochMessiah, randsymplectic, symplecticform, issymplectic
using SymplecticMatrices: williamson, Williamson, polar, Polar, blochmessiah, BlochMessiah, BlockForm, PairForm

export
    # types
    GaussianState, GaussianUnitary, GaussianChannel, GaussianLinearCombination,
    # Gaussian measurements
    generaldyne, Generaldyne, homodyne, Homodyne,
    # symplectic representations
    QuadPairBasis, QuadBlockBasis, changebasis,
    # operations
    tensor, ⊗, directsum, ⊕, apply!, ptrace, embed,
    # predefined Gaussian states
    vacuumstate, thermalstate, coherentstate, squeezedstate, eprstate,
    # non-Gaussian states
    catstate_even, catstate_odd, catstate, gkpstate,
    norm_factor,
    # predefined Gaussian channels
    displace, squeeze, twosqueeze, phaseshift, beamsplitter,
    attenuator, amplifier,
    # random objects
    randstate, randunitary, randchannel, randsymplectic,
    # wigner functions
    wigner, wignerchar,
    # symplectic form and checks
    symplecticform, issymplectic, isgaussian, sympspectrum,
    # factorizations
    williamson, Williamson, polar, Polar, blochmessiah, BlochMessiah,
    # metrics
    purity, entropy_vn, fidelity, logarithmic_negativity,
    cross_wigner, cross_wignerchar,
    # additional interface
    nmodes
    
include("errors.jl")

include("utils.jl")

include("symplectic.jl")

include("types.jl")

include("states.jl")

include("unitaries.jl")

include("channels.jl")

include("randoms.jl")

include("factorizations.jl")

include("measurements.jl")

include("generaldyne.jl")

include("homodyne.jl")

include("wigner.jl")

include("metrics.jl")

include("linearcombinations.jl")

include("nongaussian_states.jl")

end
