module Gabs

import LinearAlgebra
using LinearAlgebra: I, det, mul!, diag, qr, eigvals, Diagonal, cholesky, Symmetric, dot, Hermitian, logdet

import QuantumInterface: StateVector, AbstractOperator, apply!, tensor, ⊗, directsum, ⊕, entropy_vn, fidelity, logarithmic_negativity, ptrace

import Random
using Random: randn!

import SymplecticFactorizations: williamson, Williamson, polar, Polar, blochmessiah, BlochMessiah, randsymplectic, symplecticform, issymplectic
using SymplecticFactorizations: williamson, Williamson, polar, Polar, blochmessiah, BlochMessiah, BlockForm, PairForm

export
    # types
    GaussianState, GaussianUnitary, GaussianChannel, GaussianLinearCombination,
    # Gaussian measurements
    generaldyne, Generaldyne,
    # symplectic representations
    QuadPairBasis, QuadBlockBasis, changebasis,
    # operations
    tensor, ⊗, directsum, ⊕, apply!, ptrace,
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
    # GPU device management
    gpu, cpu, device, adapt_device,
    # optimized hybrid GPU random functions (industry standard approach)
    randstate_gpu, randunitary_gpu, randchannel_gpu, randsymplectic_gpu,
    batch_randstate_gpu, batch_randunitary_gpu,
    #GPU simulation setup
    random_ensemble_gpu, random_simulation_setup_gpu

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

include("wigner.jl")

include("metrics.jl")

include("linearcombinations.jl")

include("nongaussian_states.jl")

include("gpu_convenience.jl")
end