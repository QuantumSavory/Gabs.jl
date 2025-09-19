# Gabs.jl

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://QuantumSavory.github.io/Gabs.jl/stable)
[![](https://img.shields.io/badge/docs-dev-lightblue.svg)](https://QuantumSavory.github.io/Gabs.jl/dev)
[![Build status (Github Actions)](https://github.com/QuantumSavory/Gabs.jl/workflows/CI/badge.svg)](https://github.com/QuantumSavory/Gabs.jl/actions)
[![codecov](https://codecov.io/github/QuantumSavory/Gabs.jl/graph/badge.svg?token=JWMOD4FY6P)](https://codecov.io/github/QuantumSavory/Gabs.jl)
[![JET static analysis](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Gabs.jl is a numerical tooling package for simulating **Gaussian quantum information**.

Gaussian states and operators have the convenient property that they can be
characterized by low-dimensional matrices in the phase space representation.
Thus, a large class of continuous variable quantum information can be efficiently
simulated on a classical computer. Gabs.jl provides a high-level [Julia](https://julialang.org) interface for straightforward and high performance implementations of Gaussian quantum systems.

See the detailed [suggested readings & references page](https://QuantumSavory.github.io/Gabs.jl/dev/bibliography/) for a background on quantum information with Gaussian states.

## Installation

To install Gabs.jl, start Julia and run the following command:

```julia
using Pkg
Pkg.add("Gabs")
```
To use the package, run the command

```julia
using Gabs
```

Now, the entire library is loaded into the current workspace, with access to its
high-level interface and predefined objects.

## Example Usage

```julia
julia> basis = QuadPairBasis(1)
QuadPairBasis(1)

julia> state = vacuumstate(basis) ⊗ coherentstate(basis, 1.0-im)
GaussianState for 2 modes.
  symplectic basis: QuadPairBasis
mean: 4-element Vector{Float64}:
  0.0
  0.0
  2.0
 -2.0
covariance: 4×4 Matrix{Float64}:
 1.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0
 0.0  0.0  1.0  0.0
 0.0  0.0  0.0  1.0

julia> op = beamsplitter(basis ⊕ basis, 0.75)
GaussianUnitary for 2 modes.
  symplectic basis: QuadPairBasis
displacement: 4-element Vector{Float64}:
 0.0
 0.0
 0.0
 0.0
symplectic: 4×4 Matrix{Float64}:
  0.5        0.0       0.866025  0.0
  0.0        0.5       0.0       0.866025
 -0.866025   0.0       0.5       0.0
  0.0       -0.866025  0.0       0.5

julia> apply!(state, op)
GaussianState for 2 modes.
  symplectic basis: QuadPairBasis
mean: 4-element Vector{Float64}:
  1.7320508075688772
 -1.7320508075688772
  1.0
 -1.0
covariance: 4×4 Matrix{Float64}:
 1.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0
 0.0  0.0  1.0  0.0
 0.0  0.0  0.0  1.0

julia> ptrace(state, 1)
GaussianState for 1 mode.
  symplectic basis: QuadPairBasis
mean: 2-element Vector{Float64}:
  1.0
 -1.0
covariance: 2×2 Matrix{Float64}:
 1.0  0.0
 0.0  1.0
```
