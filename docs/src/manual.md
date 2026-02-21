# [Manual](@id Manual)

```@meta
DocTestSetup = quote
    using Gabs
end
```

Simply put, Gabs is a package for creating and transforming Gaussian bosonic systems. This section discusses the "lower level" tools for simulating such phenomena, with
mathematical explanations when appropriate. For comprehensive reviews of Gaussian
quantum information, see the [suggested readings page](@ref References).

## The Symplectic Formalism

The underlying geometry of Gaussian informatics in the phase space is *symplectic*. From the basic canonical commutation
relations (CCRs) of quantized continuous variable systems, manifestations of the symplectic group $\text{Sp}(2N, \mathbb{R})$
appear everywhere. In Gabs, symplectic basis types must be defined from the beginning. Here's how they are laid out in this library:

| canonical ordering | symplectic form | basis type |
| :---: | :---: | :---: |
| $(\hat{x}_1, \hat{p}_1, \cdots, \hat{x}_N, \hat{p}_N)$ | $\begin{pmatrix} 0 & 1 \\ - 1 & 0 \end{pmatrix} \otimes \mathbf{I}_N$ | [`QuadPairBasis`](@ref) |
| $(\hat{x}_1, \cdots, \hat{x}_N, \hat{p}_1, \cdots, \hat{p}_N)$ | $\begin{pmatrix} 0 & \mathbf{I}_N \\ -\mathbf{I}_N & 0 \end{pmatrix}$ | [`QuadBlockBasis`](@ref) |

Each symplectic basis type is wrapped around the number of bosonic modes $N$. We can compose a larger symplectic basis
with [`directsum`](@ref) or `⊕`, the direct sum symbol which can be typed in the Julia REPL as `\oplus<TAB>`:

```jldoctest
julia> b = QuadPairBasis(2)
QuadPairBasis(2)

julia> b ⊕ b
QuadPairBasis(4)
```
Of course, this type of behavior will occur implicitly when we take tensor products of Gaussian states and operators, as discussed
in the following sections.

!!! note
    A matrix $\mathbf{S}$ of size $2N\times 2N$ is symplectic when it satisfies the relation $\mathbf{S} \mathbf{\Omega} \mathbf{S}^{\text{T}} = \mathbf{\Omega},$ where $\mathbf{\Omega}$ is an invertible skew-symmetric matrix known as the *symplectic form*.

## Gaussian States

The star of this package is the [`GaussianState`](@ref) type, which allows us to initialize
and manipulate a phase space description of an arbitrary Gaussian state.

```@docs; canonical = false
GaussianState
```

Functions to create instances of elementary Gaussian states are provided as part of the package API. 
Listed below are supported predefined Gaussian states:

- [`vacuumstate`](@ref)
- [`thermalstate`](@ref)
- [`coherentstate`](@ref)
- [`squeezedstate`](@ref)
- [`eprstate`](@ref)

Detailed discussions and mathematical descriptions for each of these states are given in the
[Gaussian Zoos](@ref) page.

!!! note
    In Gabs, the default convention $\hbar = 2$ is used for the commutation relation $[\hat{x}, \hat{p}] = i\hbar$. To change this convention, pass a new value to `ħ` as a keyword argument in any predefined method that creates a Gaussian object. For instance, to change the convention to `ħ = 1` for a coherent state, call `coherentstate(basis, α, ħ = 1)`. This is a more performant and safer approach compared to setting a global variable in a package. An error will be thrown for operations between Gaussian objects with different `ħ` conventions.

## Gaussian Unitaries

To transform Gaussian states into Gaussian states, we need Gaussian maps. Let's begin with the simplest Gaussian transformation, a unitary transformation, which can be created with the [`GaussianUnitary`](@ref) type:

```@docs; canonical = false
GaussianUnitary
```

This is a rather clean way to characterize a large group of Gaussian transformations on
an `N`-mode Gaussian bosonic system. As long as we have a displacement vector of size `2N` and symplectic matrix of size `2N x 2N`, we can create a Gaussian transformation. 

This library has a number of predefined Gaussian unitaries, which are listed below:

- [`displace`](@ref)
- [`squeeze`](@ref)
- [`twosqueeze`](@ref)
- [`phaseshift`](@ref)
- [`beamsplitter`](@ref)
  
Detailed discussions and mathematical descriptions for each of these unitaries are given in the [Gaussian Zoos](@ref) page.

## Gaussian Channels

Noisy bosonic channels are an important model for describing the interaction between a Gaussian state and its environment. Similar to Gaussian unitaries, Gaussian channels are linear bosonic channels that map Gaussian states to Gaussian states. Such objects can be created with the [`GaussianChannel`](@ref) type:

```@docs; canonical = false
GaussianChannel
```

Listed below are a list of predefined Gaussian channels supported by Gabs:

- [`attenuator`](@ref)
- [`amplifier`](@ref)
  
!!! note
    Any predefined Gaussian unitary
    method can be called with an additional noise matrix to create a [`GaussianChannel`](@ref) object. For instance, a noisy displacement operator can be called with [`displace`](@ref) as follows:

    ```jldoctest
    julia> basis = QuadPairBasis(1);

    julia> noise = [1.0 -2.0; 4.0 -3.0];

    julia> displace(basis, 1.0-im, noise)
    GaussianChannel for 1 mode.
      symplectic basis: QuadPairBasis
    displacement: 2-element Vector{Float64}:
      2.0
     -2.0
    transform: 2×2 Matrix{Float64}:
     1.0  0.0
     0.0  1.0
    noise: 2×2 Matrix{Float64}:
     1.0  -2.0
     4.0  -3.0
    ```

## Actions

Out-of-place actions of Gaussian unitaries and Gaussian channels on Gaussian states
are called with `*`, while in-place ones are called with [`apply!`](@ref):

```jldoctest
julia> basis = QuadBlockBasis(2); state = vacuumstate(basis);

julia> un = squeeze(basis, 1.0, 2.0)
GaussianUnitary for 2 modes.
  symplectic basis: QuadBlockBasis
displacement: 4-element Vector{Float64}:
 0.0
 0.0
 0.0
 0.0
symplectic: 4×4 Matrix{Float64}:
  2.03214   0.0      -1.06861   0.0
  0.0       2.03214   0.0      -1.06861
 -1.06861   0.0       1.05402   0.0
  0.0      -1.06861   0.0       1.05402

julia> un * state
GaussianState for 2 modes.
  symplectic basis: QuadBlockBasis
mean: 4-element Vector{Float64}:
 0.0
 0.0
 0.0
 0.0
covariance: 4×4 Matrix{Float64}:
  5.2715    0.0      -3.29789   0.0
  0.0       5.2715    0.0      -3.29789
 -3.29789   0.0       2.25289   0.0
  0.0      -3.29789   0.0       2.25289

julia> ch = attenuator(basis, 0.25, 5)
GaussianChannel for 2 modes.
  symplectic basis: QuadBlockBasis
displacement: 4-element Vector{Float64}:
 0.0
 0.0
 0.0
 0.0
transform: 4×4 Matrix{Float64}:
 0.968912  0.0       0.0       0.0
 0.0       0.968912  0.0       0.0
 0.0       0.0       0.968912  0.0
 0.0       0.0       0.0       0.968912
noise: 4×4 Matrix{Float64}:
 0.306044  0.0       0.0       0.0
 0.0       0.306044  0.0       0.0
 0.0       0.0       0.306044  0.0
 0.0       0.0       0.0       0.306044

julia> apply!(state, ch)
GaussianState for 2 modes.
  symplectic basis: QuadBlockBasis
mean: 4-element Vector{Float64}:
 0.0
 0.0
 0.0
 0.0
covariance: 4×4 Matrix{Float64}:
 1.24483  0.0      0.0      0.0
 0.0      1.24483  0.0      0.0
 0.0      0.0      1.24483  0.0
 0.0      0.0      0.0      1.24483
```

## Tensor Products

If we were operating in the state (Fock) space, and wanted to describe multi-mode Gaussian states,
we would take the tensor product of multiple density operators. That method, however,
is quite computationally expensive and requires a finite truncation of the Fock basis. To create
such state vector simulations, we recommend using the [QuantumOptics.jl](https://github.com/qojulia/QuantumOptics.jl) library. For our purposes in the phase space, we efficiently create multi-mode Gaussian systems via direct sum, which corresponds to a tensor product of infinite-dimensional Hilbert spaces. A tensor product of Gaussian states can be called with either [`tensor`](@ref) or `⊗`, the Kronecker product symbol
which can be typed in the Julia REPL as `\otimes<TAB>`. Take the following example, where we produce a 3-mode Gaussian state that consists of a coherent state, vacuumstate, and squeezed state:

```jldoctest
julia> basis = QuadPairBasis(1);

julia> coherentstate(basis, -1.0+im) ⊗ vacuumstate(basis) ⊗ squeezedstate(basis, 0.25, pi/4)
GaussianState for 3 modes.
  symplectic basis: QuadPairBasis
mean: 6-element Vector{Float64}:
 -2.0
  2.0
  0.0
  0.0
  0.0
  0.0
covariance: 6×6 Matrix{Float64}:
 1.0  0.0  0.0  0.0   0.0        0.0
 0.0  1.0  0.0  0.0   0.0        0.0
 0.0  0.0  1.0  0.0   0.0        0.0
 0.0  0.0  0.0  1.0   0.0        0.0
 0.0  0.0  0.0  0.0   0.759156  -0.36847
 0.0  0.0  0.0  0.0  -0.36847    1.4961
```

Note that in the above example, we defined the symplectic basis to be of type [`QuadPairBasis`](@ref). If we wanted the canonical field operators to be ordered blockwise, then we would call [`QuadBlockBasis`](@ref) instead:

```jldoctest
julia> basis = QuadBlockBasis(1);

julia> coherentstate(basis, -1.0+im) ⊗ vacuumstate(basis) ⊗ squeezedstate(basis, 0.25, pi/4)
GaussianState for 3 modes.
  symplectic basis: QuadBlockBasis
mean: 6-element Vector{Float64}:
 -2.0
  0.0
  0.0
  2.0
  0.0
  0.0
covariance: 6×6 Matrix{Float64}:
 1.0  0.0   0.0       0.0  0.0   0.0
 0.0  1.0   0.0       0.0  0.0   0.0
 0.0  0.0   0.759156  0.0  0.0  -0.36847
 0.0  0.0   0.0       1.0  0.0   0.0
 0.0  0.0   0.0       0.0  1.0   0.0
 0.0  0.0  -0.36847   0.0  0.0   1.4961
```
These tensor product methods are also available for Gaussian unitaries and channels:

```jldoctest
julia> basis = QuadBlockBasis(1);

julia> squeeze(basis, 2.0, pi/3) ⊗ phaseshift(basis, pi/6)
GaussianUnitary for 2 modes.
  symplectic basis: QuadBlockBasis
displacement: 4-element Vector{Float64}:
 0.0
 0.0
 0.0
 0.0
symplectic: 4×4 Matrix{Float64}:
  1.94877   0.0       -3.14095  0.0
  0.0       0.866025   0.0      0.5
 -3.14095   0.0        5.57563  0.0
  0.0      -0.5        0.0      0.866025
```

For applying the same predefined operator to a multi-mode system, simply call
the operator on the corresponding multi-mode basis. For instance, if we wanted to
apply a phase shift of `π/4` to a three-mode Gaussian system, then we would
create the following operation:

```jldoctest
julia> basis = QuadPairBasis(3);

julia> phaseshift(basis, pi/4)
GaussianUnitary for 3 modes.
  symplectic basis: QuadPairBasis
displacement: 6-element Vector{Float64}:
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
symplectic: 6×6 Matrix{Float64}:
  0.707107  0.707107   0.0       0.0        0.0       0.0
 -0.707107  0.707107   0.0       0.0        0.0       0.0
  0.0       0.0        0.707107  0.707107   0.0       0.0
  0.0       0.0       -0.707107  0.707107   0.0       0.0
  0.0       0.0        0.0       0.0        0.707107  0.707107
  0.0       0.0        0.0       0.0       -0.707107  0.707107
```

If, instead we wanted to apply phase shifts of `π/3`, `π/4`, and `π/5`
to the respective-modes of a three-mode Gaussian system, then we would dispatch
`phaseshift` on a vector of the phase shifts:

```jldoctest
julia> basis = QuadPairBasis(3);

julia> phaseshift(basis, [pi/3, pi/4, pi/5])
GaussianUnitary for 3 modes.
  symplectic basis: QuadPairBasis
displacement: 6-element Vector{Float64}:
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
symplectic: 6×6 Matrix{Float64}:
  0.5       0.866025   0.0       0.0        0.0       0.0
 -0.866025  0.5        0.0       0.0        0.0       0.0
  0.0       0.0        0.707107  0.707107   0.0       0.0
  0.0       0.0       -0.707107  0.707107   0.0       0.0
  0.0       0.0        0.0       0.0        0.809017  0.587785
  0.0       0.0        0.0       0.0       -0.587785  0.809017
```

Similar properties hold for Gaussian channels and states. Let's see some examples
for multi-mode coherent states:

```jldoctest
julia> basis = QuadPairBasis(3);

julia> coherentstate(basis, 1.0-im)
GaussianState for 3 modes.
  symplectic basis: QuadPairBasis
mean: 6-element Vector{Float64}:
  2.0
 -2.0
  2.0
 -2.0
  2.0
 -2.0
covariance: 6×6 Matrix{Float64}:
 1.0  0.0  0.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0  0.0  0.0
 0.0  0.0  1.0  0.0  0.0  0.0
 0.0  0.0  0.0  1.0  0.0  0.0
 0.0  0.0  0.0  0.0  1.0  0.0
 0.0  0.0  0.0  0.0  0.0  1.0

julia> coherentstate(basis, [1.0-im, 2.0-2.0im, 3.0-3.0im])
GaussianState for 3 modes.
  symplectic basis: QuadPairBasis
mean: 6-element Vector{Float64}:
  2.0
 -2.0
  4.0
 -4.0
  6.0
 -6.0
covariance: 6×6 Matrix{Float64}:
 1.0  0.0  0.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0  0.0  0.0
 0.0  0.0  1.0  0.0  0.0  0.0
 0.0  0.0  0.0  1.0  0.0  0.0
 0.0  0.0  0.0  0.0  1.0  0.0
 0.0  0.0  0.0  0.0  0.0  1.0
```

## Partial Traces

Partial traces of Gaussian states can be performed with [`ptrace`](@ref). For tracing 
out a single-mode, call an integer corresponding to the mode of choice in a multi-mode Gaussian system. For tracing out several modes, call instead a vector of integers. 
Let's see some examples:
```jldoctest
julia> basis = QuadPairBasis(2);

julia> state = coherentstate(basis, [1.0-im, 2.0-2.0im]) ⊗ eprstate(basis, 2.0, pi/3)
GaussianState for 4 modes.
  symplectic basis: QuadPairBasis
mean: 8-element Vector{Float64}:
  2.0
 -2.0
  4.0
 -4.0
  0.0
  0.0
  0.0
  0.0
covariance: 8×8 Matrix{Float64}:
 1.0  0.0  0.0  0.0    0.0       0.0       0.0       0.0
 0.0  1.0  0.0  0.0    0.0       0.0       0.0       0.0
 0.0  0.0  1.0  0.0    0.0       0.0       0.0       0.0
 0.0  0.0  0.0  1.0    0.0       0.0       0.0       0.0
 0.0  0.0  0.0  0.0   27.3082    0.0     -13.645   -23.6338
 0.0  0.0  0.0  0.0    0.0      27.3082  -23.6338   13.645
 0.0  0.0  0.0  0.0  -13.645   -23.6338   27.3082    0.0
 0.0  0.0  0.0  0.0  -23.6338   13.645     0.0      27.3082

julia> ptrace(state, 1)
GaussianState for 3 modes.
  symplectic basis: QuadPairBasis
mean: 6-element Vector{Float64}:
  4.0
 -4.0
  0.0
  0.0
  0.0
  0.0
covariance: 6×6 Matrix{Float64}:
 1.0  0.0    0.0       0.0       0.0       0.0
 0.0  1.0    0.0       0.0       0.0       0.0
 0.0  0.0   27.3082    0.0     -13.645   -23.6338
 0.0  0.0    0.0      27.3082  -23.6338   13.645
 0.0  0.0  -13.645   -23.6338   27.3082    0.0
 0.0  0.0  -23.6338   13.645     0.0      27.3082

julia> ptrace(state, [1, 4])
GaussianState for 2 modes.
  symplectic basis: QuadPairBasis
mean: 4-element Vector{Float64}:
  4.0
 -4.0
  0.0
  0.0
covariance: 4×4 Matrix{Float64}:
 1.0  0.0   0.0      0.0
 0.0  1.0   0.0      0.0
 0.0  0.0  27.3082   0.0
 0.0  0.0   0.0     27.3082
```

## Symplectic Analysis

Gabs provides different tools for analyzing symplectic transformations and properties
of Gaussian states and operators. Under the hood, the aforementioned types such 
as [`GaussianState`](@ref), [`GaussianUnitary`](@ref), and [`GaussianChannel`](@ref)
keep track of symplectic bases, i.e., ordering of bosonic mode operators. To change
symplectic bases, simply call [`changebasis`](@ref). As an example, consider the phase shift operator, defined by the quadrature transformations
```math
\hat{x} \to \cos(\theta) \hat{x} + \sin(\theta) \hat{p}, \qquad \hat{p} \to -\sin(\theta) \hat{x} + \cos(\theta) \hat{p}.
```
The symplectic transformation corresponding to the action of a tensor product of phase shift operators
on a bosonic system is dependent on the ordering of the bosonic mode observables, so it is useful to swap
symplectic bases. Consider a simple example for a two-mode system:
```jldoctest
julia> op = phaseshift(QuadBlockBasis(2), 0.5)
GaussianUnitary for 2 modes.
  symplectic basis: QuadBlockBasis
displacement: 4-element Vector{Float64}:
 0.0
 0.0
 0.0
 0.0
symplectic: 4×4 Matrix{Float64}:
  0.877583   0.0       0.479426  0.0
  0.0        0.877583  0.0       0.479426
 -0.479426   0.0       0.877583  0.0
  0.0       -0.479426  0.0       0.877583

julia> changebasis(QuadPairBasis, op)
GaussianUnitary for 2 modes.
  symplectic basis: QuadPairBasis
displacement: 4-element Vector{Float64}:
 0.0
 0.0
 0.0
 0.0
symplectic: 4×4 Matrix{Float64}:
  0.877583  0.479426   0.0       0.0
 -0.479426  0.877583   0.0       0.0
  0.0       0.0        0.877583  0.479426
  0.0       0.0       -0.479426  0.877583
```
Various symplectic decompositions are supported in Gabs through the symplectic linear algebra package [SymplecticFactorizations.jl](https://github.com/apkille/SymplecticFactorizations.jl). Particularly important
ones are the Williamson decomposition ([`williamson`](@ref)), Bloch-Messiah/Euler decomposition ([`blochmessiah`](@ref)), and the symplectic polar decomposition ([`polar`](@ref)):
```@docs; canonical = false
williamson
```
```@docs; canonical = false
blochmessiah
```
```@docs; canonical = false
polar
```
Let's see an example with the Williamson decomposition:
```@repl; using Gabs
using Gabs, LinearAlgebra
state = randstate(QuadBlockBasis(1))
F = williamson(state)
isapprox(Diagonal(repeat(F.spectrum, 2)), F.S * state.covar * F.S', atol = 1e-12)
S, spectrum = F; # destructuring via iteration
S == F.S && spectrum == F.spectrum
issymplectic(QuadBlockBasis(1), S, atol = 1e-12)
```
In the last line of code, we used the symplectic check [`issymplectic`](@ref). In general, we can
check if a state or operator is Gaussian with [`isgaussian`](@ref).