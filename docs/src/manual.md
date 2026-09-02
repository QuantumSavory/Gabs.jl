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

To apply a Gaussian unitary or channel to selected modes, place the mode index or
indices before the operation: `apply!(state, indices, op)`.

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
```@repl
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

## Stellar States

Every Gaussian state is pure or mixed, but no Gaussian state exhibits negativity in its Wigner function. To describe pure non-Gaussian states while keeping the phase space formalism, Gabs provides the [`StellarState`](@ref) type, which stores a state in the form $|\psi\rangle = \hat{U}|C\rangle$, where $\hat{U}$ is a Gaussian unitary and $|C\rangle$ is a normalized superposition of Fock states with finite support.

```@docs; canonical = false
StellarState
```

The core is stored as an array with one index per mode, so the amplitude of $|k_1, \ldots, k_N\rangle$ sits at `core[k₁+1, …, k_N+1]`. The Gaussian factor is a [`GaussianUnitary`](@ref), and it supplies the symplectic basis and the `ħ` convention of the stellar state, so `x.basis` and `x.ħ` are forwarded to it.

The following predefined functions are supported for creating `StellarState` objects:
- [`fockstate`](@ref)
- [`randstellar`](@ref)

A Fock state is an elementary case, with an identity Gaussian factor and a core supported on a single multi-index:

```jldoctest
julia> fockstate(QuadPairBasis(1), 2)
StellarState for 1 mode.
  symplectic basis: QuadPairBasis
  stellar rank: 2
core: 3-element Vector{ComplexF64}:
 0.0 + 0.0im
 0.0 + 0.0im
 1.0 + 0.0im
displacement: 2-element Vector{Float64}:
 0.0
 0.0
symplectic: 2×2 Matrix{Float64}:
 1.0  0.0
 0.0  1.0
```

As with [`coherentstate`](@ref), a scalar argument is applied to every mode, while a vector assigns an occupation to each mode individually:

```jldoctest
julia> basis = QuadPairBasis(3);

julia> stellarrank(fockstate(basis, 2))
6

julia> stellarrank(fockstate(basis, [1, 0, 2]))
3
```

In this formalism, photon addition and subtraction both reduce to a single question: what does a ladder
operator become when it is pushed through the Gaussian factor? For a Gaussian
unitary characterized by displacement vector $\mathbf{d}$, symplectic matrix $\mathbf{S}$,
and mode operators $\hat{a}_j = (\hat{x}_j + i \hat{p}_j)/\sqrt{2\hbar}$, we have the
transformation

```math
\hat{U}^{\dagger} \hat{a}_j^{\dagger} \hat{U} = \sum_{k} \left( \mu_{jk} \hat{a}_k^{\dagger} + \nu_{jk} \hat{a}_k \right) + \gamma_j,
```

where $\boldsymbol{\mu}$ and $\boldsymbol{\nu}$ are built from the four blocks of
$\mathbf{S}$ and $\boldsymbol{\gamma}$ from $\mathbf{d}$, in the same convention
$\hat{U}^{\dagger} \hat{\boldsymbol{\xi}} \hat{U} = \mathbf{S} \hat{\boldsymbol{\xi}} + \mathbf{d}$
that [`GaussianUnitary`](@ref) stores. Symplecticity of $\mathbf{S}$ is equivalent to the
relations $\boldsymbol{\mu}\boldsymbol{\mu}^{\dagger} - \boldsymbol{\nu}\boldsymbol{\nu}^{\dagger} = \mathbf{I}$
and $\boldsymbol{\mu}\boldsymbol{\nu}^{\text{T}} = \boldsymbol{\nu}\boldsymbol{\mu}^{\text{T}}$
being satisfied, the first of which shows that $\boldsymbol{\mu}$ is invertible. The
Gaussian factor is passive when $\boldsymbol{\nu} = 0$.

Its a fun little exercise to check the above formula against your favorite unitary and see how the machinery works. Take, for example the definitions in [`squeeze`](@ref) or [`displace`](@ref).

!!! note
    The decomposition $|\psi\rangle = \hat{U}|C\rangle$ is not unique. For any passive unitary $\hat{V}$, the pair $(\hat{V}|C\rangle, \hat{U}\hat{V}^{\dagger})$ describes the same state, since passive unitaries are the ones that preserve finite Fock support. Thus equality and `Base.isapprox` for `StellarState` compare the stored core and Gaussian factor, not the states they represent.

### Stellar Rank

The stellar rank $r$ of $|\psi\rangle = \hat{U}|C\rangle$ is the total degree of the core, i.e. the largest $|\mathbf{k}| = k_1 + \cdots + k_N$ carrying a nonzero amplitude. It is computed with [`stellarrank`](@ref). Two properties make it the natural measure of non-Gaussianity in this setting: it is invariant under every Gaussian unitary, and it vanishes exactly on the pure Gaussian states.

```jldoctest
julia> basis = QuadPairBasis(1); x = fockstate(basis, 1);

julia> stellarrank(squeeze(basis, 1.0, pi/4) * x)
1

julia> isgaussian(x)
false

julia> isgaussian(fockstate(basis, 0))
true
```

Hence, Gabs uses the following rules for type conversion. A pure [`GaussianState`](@ref) becomes a rank-zero stellar state whose Gaussian factor is the positive symmetric symplectic square root of $(2/\hbar)\mathbf{V}$, and a rank-zero stellar state becomes the Gaussian state obtained by applying its Gaussian factor to the vacuum:

```jldoctest
julia> basis = QuadPairBasis(1);

julia> x = StellarState(squeezedstate(basis, 0.5, pi/4));

julia> stellarrank(x)
0

julia> GaussianState(x) ≈ squeezedstate(basis, 0.5, pi/4)
true
```

Conversion in either direction throws when the hypothesis fails: `StellarState` requires the Gaussian state to be pure, and `GaussianState` requires the stellar rank to vanish.

Actions of Gaussian unitaries are called with `*` and [`apply!`](@ref), exactly as for Gaussian states, and act on the Gaussian factor alone while leaving the core untouched. Tensor products, [`embed`](@ref), and [`changebasis`](@ref) are likewise supported, and the rank is additive under `⊗`:

```jldoctest
julia> basis = QuadPairBasis(1);

julia> fockstate(basis, 1) ⊗ fockstate(basis, 2) ≈ fockstate(QuadPairBasis(2), [1, 2])
true

julia> stellarrank(fockstate(basis, 1) ⊗ fockstate(basis, 2))
3
```

!!! note
    [`ptrace`](@ref) is not defined for stellar states. Discarding a mode of an entangled pure state produces a mixed state, which admits no decomposition of the form $\hat{U}|C\rangle$. For the same reason [`purity`](@ref) returns `1` and [`entropy_vn`](@ref) returns `0` for every `StellarState`.

### Photon Addition and Subtraction

Normalized photon addition and subtraction on a chosen mode are provided by [`addphoton`](@ref) and [`subtractphoton`](@ref). Both leave the Gaussian factor untouched and push the ladder operator through it, so the core absorbs $\hat{U}^{\dagger} \hat{a}_i^{\dagger} \hat{U}$ or $\hat{U}^{\dagger} \hat{a}_i \hat{U}$, whose coefficients are the $i$-th rows of the Bogoliubov equation written above.

```jldoctest
julia> basis = QuadPairBasis(1); vac = fockstate(basis, 0);

julia> addphoton(vac) ≈ fockstate(basis, 1)
true

julia> stellarrank(addphoton(addphoton(vac)))
2
```

Addition raises the rank by exactly one, since the top-degree part of the core is multiplied by the linear form $\sum_k \mu_{ik} z_k$, which is nonzero because $\boldsymbol{\mu}$ is invertible. Subtraction is not symmetric with it, and its effect on the rank is governed by row $i$ of $\boldsymbol{\nu}$ and by $\gamma_i$:

| Gaussian factor at mode `index` | example | rank of `subtractphoton(x)` |
| :---: | :---: | :---: |
| $\nu_{i\cdot} \neq 0$ | squeezed vacuum | $r+1$ |
| $\nu_{i\cdot} = 0$, $\gamma_i \neq 0$ | displaced Fock state | $r$ |
| $\nu_{i\cdot} = 0$, $\gamma_i = 0$ | Fock state | $r-1$ |

```jldoctest
julia> basis = QuadPairBasis(1); vac = fockstate(basis, 0);

julia> stellarrank(subtractphoton(squeeze(basis, 1.0, 0.0) * vac))
1

julia> stellarrank(subtractphoton(displace(basis, 1.0+im) * fockstate(basis, 1)))
1

julia> stellarrank(subtractphoton(fockstate(basis, 1)))
0
```

The middle row is the statement that a coherent state is an eigenstate of the annihilation operator. That is, subtraction returns the same state up to normalization, and the rank is unchanged. The last row is the only case in which the core can shrink, and subtraction throws when it would annihilate the state, as for $\hat{a}|0\rangle$.

### The Stellar Function

The stellar (also called Bargmann), function of a state with Fock amplitudes $\psi_{\mathbf{k}}$ is

```math
F_{\psi}(\mathbf{z}) = \sum_{\mathbf{k}} \psi_{\mathbf{k}} \frac{\mathbf{z}^{\mathbf{k}}}{\sqrt{\mathbf{k}!}}, \quad \text{where} \quad \mathbf{z}^{\mathbf{k}} = \prod_j z_j^{k_j}, \quad \mathbf{k}! = \prod_j k_j!,
```

and is evaluated with [`stellarfunction`](@ref).

```@docs; canonical = false
stellarfunction
```

For a stellar state the sum has a closed form. Pushing the core through the Gaussian factor gives

```math
F_{\psi}(\mathbf{z}) = \mathcal{N} \exp\left( \tfrac{1}{2} \mathbf{z}^{\text{T}} \mathbf{\Sigma} \mathbf{z} + \boldsymbol{\tau}^{\text{T}} \mathbf{z} \right) p(\mathbf{z}), \qquad \mathbf{\Sigma} = \bar{\boldsymbol{\nu}} \boldsymbol{\mu}^{-1}, \quad \boldsymbol{\tau} = \bar{\boldsymbol{\gamma}} - \mathbf{\Sigma} \boldsymbol{\gamma},
```

with $p$ a polynomial of total degree $r$. The Gaussian prefactor never vanishes, so the zeros of $F_{\psi}$ are the zeros of $p$. For a single mode this is the origin of the name: the stellar rank counts the zeros of the stellar function in the complex plane, with multiplicity.

```jldoctest
julia> basis = QuadPairBasis(1); α = 1.0 + 0.5im;

julia> stellarfunction(displace(basis, α) * fockstate(basis, 0), 0.4 + 0.2im)
0.6654917625398412 + 0.2813654043279519im

julia> stellarfunction(displace(basis, α) * fockstate(basis, 1), conj(α))
0.0 + 0.0im
```

The first state is coherent, with $F(z) = e^{-|\alpha|^2/2 + \alpha z}$, which is nowhere zero and has rank zero. The second is a displaced single-photon state, with $F(z) = (z - \bar{\alpha}) e^{-|\alpha|^2/2 + \alpha z}$, whose single zero sits at $\bar{\alpha}$ and accounts for its rank of one. A squeezed vacuum, $F(z) = e^{-\tanh(r) z^2/2}/\sqrt{\cosh r}$, is again nowhere zero.

!!! note
    A [`GaussianUnitary`](@ref) records the pair $(\mathbf{d}, \mathbf{S})$ rather than a metaplectic operator, so the Gaussian factor determines $\hat{U}$ only up to a global phase and $F_{\psi}$ is fixed only up to the same phase. The gauge used here is $\mathcal{N} > 0$.