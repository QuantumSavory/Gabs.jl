# [Manual](@id Manual)

```@meta
DocTestSetup = quote
    using Gabs
end
```

Simply put, Gabs is a package for creating and transforming Gaussian bosonic systems. This section discusses the "lower level" tools for simulating such phenomena, with
mathematical explanations when appropriate. For comprehensive reviews of Gaussian
quantum information, see the [suggested readings page](@ref References).

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

If we were operating in the state (Fock) space, and wanted to describe multi-mode Gaussian states,
we would take the tensor product of multiple density operators. That method, however,
is quite computationally expensive and requires a finite truncation of the Fock basis. To create
such state vector simulations, we recommend using the [QuantumOptics.jl](https://github.com/qojulia/QuantumOptics.jl) library. For our purposes in the phase space, we can manually create multi-mode Gaussian systems with a tensor product, which can be called with either [`tensor`](@ref) or `⊗`, the Kronecker product symbol
which can be typed in the Julia REPL as `\otimes<TAB>`. Take the following example, where we produce a 3-mode Gaussian state that consists of a coherent state, vacuumstate, and squeezed state:

```jldoctest
julia> basis = QuadPairBasis(1);

julia> coherentstate(basis, -1.0+im) ⊗ vacuumstate(basis) ⊗ squeezedstate(basis, 0.25, pi/4)
GaussianState for 3 modes in QuadPairBasis representation.
mean: 6-element Vector{Float64}:
 -1.4142135623730951
  1.4142135623730951
  0.0
  0.0
  0.0
  0.0
covariance: 6×6 Matrix{Float64}:
 1.0  0.0  0.0  0.0  0.0       0.0
 0.0  1.0  0.0  0.0  0.0       0.0
 0.0  0.0  1.0  0.0  0.0       0.0
 0.0  0.0  0.0  1.0  0.0       0.0
 0.0  0.0  0.0  0.0  0.379578  0.184235
 0.0  0.0  0.0  0.0  0.184235  0.748048
```

Note that in the above example, we defined the symplectic basis of the Gaussian state to be [`QuadPairBasis`](@ref), which determines the arrangement of our quadrature field operators to be pairwise: $\mathbf{\hat{x}} = (q_1, p_1, q_2, p_2, q_3, p_3)^{\text{T}}$. If we wanted the field operators to be ordered blockwise, i.e.,
$\mathbf{\hat{x}} = (q_1, q_2, q_3, p_1, p_2, p_3)^{\text{T}}$ then we would call [`QuadBlockBasis`](@ref) instead:

```julia
julia> basis = QuadBlockBasis(1);

julia> coherentstate(basis, -1.0+im) ⊗ vacuumstate(basis) ⊗ squeezedstate(basis, 0.25, pi/4)
GaussianState for 3 modes in QuadBlockBasis representation.
mean: 6-element Vector{Float64}:
 -1.4142135623730951
  0.0
  0.0
  1.4142135623730951
  0.0
  0.0
covariance: 6×6 Matrix{Float64}:
 1.0  0.0  0.0       0.0  0.0  0.0
 0.0  1.0  0.0       0.0  0.0  0.0
 0.0  0.0  0.379578  0.0  0.0  0.184235
 0.0  0.0  0.0       1.0  0.0  0.0
 0.0  0.0  0.0       0.0  1.0  0.0
 0.0  0.0  0.184235  0.0  0.0  0.748048
```

## Gaussian Operators

To transform Gaussian states into Gaussian states, we need Gaussian maps. There are various ways to construct Gaussian transformations, which we will discuss in this section.

### Gaussian Unitaries

Let's begin with the simplest Gaussian transformation, a unitary transformation, which can be created with the [`GaussianUnitary`](@ref) type:

```@docs; canonical = false
GaussianUnitary
```

This is a rather clean way to characterize a large group of Gaussian transformations on
an `N`-mode Gaussian bosonic system. As long as we have a displacement vector of size `2N` and symplectic matrix of size `2N x 2N`, we can create a Gaussian transformation. 

!!! note
    A matrix $\mathbf{S}$ of size $2N\times 2N$ is symplectic when it satisfies the following relation:

    $$\mathbf{S} \mathbf{\Omega} \mathbf{S}^{\text{T}} = \mathbf{\Omega}.

    In this library, we define symplectic matrices with respect to $\Omega$, the *symplectic form*, which satisfies the canonical
    commutation relation $[\hat{x}_i, \hat{x}_j] = 2i\Omega_{ij}$, where $\hat{x}_i$ and $\hat{x}_j$
    are quadrature field operators.

This library has a number of predefined Gaussian unitaries, which are listed below:

- [`displace`](@ref)
- [`squeeze`](@ref)
- [`twosqueeze`](@ref)
- [`phaseshift`](@ref)
- [`beamsplitter`](@ref)
  
Detailed discussions and mathematical descriptions for each of these unitaries are given in the [Gaussian Zoos](@ref) page.

### Gaussian Channels

Noisy bosonic channels are an important model for describing the interaction between a Gaussian state and its environment. Similar to Gaussian unitaries, Gaussian channels are linear bosonic channels that map Gaussian states to Gaussian states. Such objects can be created with the [`GaussianChannel`](@ref) type:

```@docs; canonical = false
GaussianChannel
```

Listed below are a list of predefined Gaussian channels supported by Gabs:

- [`attenuator`](@ref)
- [`amplifier`](@ref)
  
!!! note
    When its noise matrix $\mathbf{N} = \mathbf{0}$ and transform operator $\mathbf{T}$ is a symplectic matrix, a Gaussian channel is a unitary operator. Any predefined Gaussian unitary
    method can be called with an additional noise matrix to create a [`GaussianChannel`](@ref) object. For instance, a noisy displacement operator can be called with [`displace`](@ref) as follows:

    ```jldoctest
    julia> basis = QuadPairBasis(1);

    julia> noise = [1.0 -2.0; 4.0 -3.0];

    julia> displace(basis, 1.0-im, noise)
    GaussianChannel for 1 mode in QuadPairBasis representation.
    displacement: 2-element Vector{Float64}:
      1.4142135623730951
     -1.4142135623730951
    transform: 2×2 Matrix{Float64}:
     1.0  0.0
     0.0  1.0
    noise: 2×2 Matrix{Float64}:
     1.0  -2.0
     4.0  -3.0
    ```