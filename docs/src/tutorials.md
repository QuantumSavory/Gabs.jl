# [Tutorials](@id Tutorials)

```@meta
DocTestSetup = quote
    using Gabs
end
```

## Visualizations

Quasiprobability distributions of Gaussian states can be visualized with [Makie.jl](https://github.com/MakieOrg/Makie.jl). Gabs.jl
currently has support for the following distributions, which can be called with the keyword argument `dist`:
- [`wigner`](@ref)
- [`wignerchar`](@ref)

Below is a code example that plots the wigner distribution of a vacuum state:
```@example
using Gabs, CairoMakie
basis = QuadPairBasis(1)
state = vacuumstate(basis)
q, p = collect(-5.0:0.5:5.0), collect(-5.0:0.5:5.0) # phase space coordinates
heatmap(q, p, state, dist = :wigner)
```
Of course, more plotting sugar can be added to this example with internal Makie attributes:
```@example
using Gabs, CairoMakie
basis = QuadPairBasis(1)
state = vacuumstate(basis)
q, p = collect(-5.0:0.5:5.0), collect(-5.0:0.5:5.0) # phase space coordinates
fig = Figure(fontsize=15, size = (375, 300), fonts = (; regular="CMU Serif"))
ax = Axis(fig[1,1], xlabel = L"q", ylabel = L"p")
hm = heatmap!(ax, q, p, state, dist = :wigner, colormap = :heat)
Colorbar(fig[1,2], hm)
fig
```
## Using Custom Arrays

Types such as [`GaussianState`](@ref), [`GaussianUnitary`](@ref), and
[`GaussianChannel`](@ref) are agnostic array wrappers, so they can hold any custom array that exists
in the Julia ecosystem. Operations in Gabs.jl preserve these custom array types, provided they
follow the [`AbstractArray` interface](https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array).

Let's explore this feature in detail, using [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) and [SparseArrays.jl](https://github.com/JuliaSparse/SparseArrays.jl) as a case study. To create a coherent state that wraps around pure Julia arrays, the function
[`coherentstate`](@ref) can be called with a single complex argument:
```jldoctest
julia> coherentstate(QuadPairBasis(1), 1.0-im)
GaussianState for 1 mode.
  symplectic basis: QuadPairBasis
mean: 2-element Vector{Float64}:
  2.0
 -2.0
covariance: 2×2 Matrix{Float64}:
 1.0  0.0
 0.0  1.0
```
Here, a [`GaussianState`](@ref) object containing a `Vector{Float64}` of size two (mean vector)
and a `Matrix{Float64}` of size two by two (covariance matrix) was initialized. If we want to potentially optimize and boost the performance of our code,
then one approach would be to wrap our [`GaussianState`](@ref) around different array types, for instance, an [`SVector`](https://juliaarrays.github.io/StaticArrays.jl/stable/api/#StaticArraysCore.SVector) from [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) or a [`SparseVector`](https://docs.julialang.org/en/v1/stdlib/SparseArrays/#SparseArrays.SparseVector) from [SparseArrays.jl](https://github.com/JuliaSparse/SparseArrays.jl). Any defined method in Gabs.jl that (i) creates a custom type or (ii) transforms a custom type
can specify an array type in its first (and second) arguments. Let's see an example with [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl):
```jldoctest
julia> using StaticArrays

julia> state = coherentstate(SVector{2}, SMatrix{2,2}, QuadPairBasis(1), 1.0-im)
GaussianState for 1 mode.
  symplectic basis: QuadPairBasis
mean: 2-element SVector{2, Float64} with indices SOneTo(2):
  2.0
 -2.0
covariance: 2×2 SMatrix{2, 2, Float64, 4} with indices SOneTo(2)×SOneTo(2):
 1.0  0.0
 0.0  1.0

julia> tp = state ⊗ state
GaussianState for 2 modes.
  symplectic basis: QuadPairBasis
mean: 4-element SVector{4, Float64} with indices SOneTo(4):
  2.0
 -2.0
  2.0
 -2.0
covariance: 4×4 SMatrix{4, 4, Float64, 16} with indices SOneTo(4)×SOneTo(4):
 1.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0
 0.0  0.0  1.0  0.0
 0.0  0.0  0.0  1.0

julia> using SparseArrays

julia> ptrace(SparseVector, SparseMatrixCSC, tp, 1)
GaussianState for 1 mode.
  symplectic basis: QuadPairBasis
mean: 2-element SparseVector{Float64, Int64} with 2 stored entries:
  [1]  =  2.0
  [2]  =  -2.0
covariance: 2×2 SparseMatrixCSC{Float64, Int64} with 2 stored entries:
 1.0   ⋅ 
  ⋅   1.0
```
Importantly, methods that create or manipulate a Gaussian state, such as [`tensor`](@ref) and [`ptrace`](@ref), preserve array types, but can also opt for a different array type.

!!! note
    If you have an array wrapper that initializes both vectors
    and matrices, then you can specify the array type with a single argument. For instance, to initialize a state that contains `Array`s holding numbers of type `Float32` rather
    than `Float64`, simply pass `Array{Float32}` to any relevant Gabs.jl method:
    ```jldoctest
    julia> state = displace(Array{Float32}, QuadPairBasis(1), 1.0-im)
    GaussianUnitary for 1 mode.
      symplectic basis: QuadPairBasis
    displacement: 2-element Vector{Float32}:
      2.0
     -2.0
    symplectic: 2×2 Matrix{Float32}:
     1.0  0.0
     0.0  1.0
    ```

## Creating Symbolic Gaussian States

Create Gaussian states with symbolic variables using [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl):

```jldoctest
julia> using Symbolics

julia> @variables r θ τ
3-element Vector{Num}:
 r
 θ
 τ

julia> b = QuadBlockBasis(2);

julia> st = eprstate(b, r, θ)
GaussianState for 2 modes.
  symplectic basis: QuadBlockBasis
mean: 4-element Vector{Num}:
 0
 0
 0
 0
covariance: 4×4 Matrix{Num}:
         cosh(2r)  -sinh(2r)*cos(θ)                 0  -sinh(2r)*sin(θ)
 -sinh(2r)*cos(θ)          cosh(2r)  -sinh(2r)*sin(θ)                 0
                0  -sinh(2r)*sin(θ)          cosh(2r)   sinh(2r)*cos(θ)
 -sinh(2r)*sin(θ)                 0   sinh(2r)*cos(θ)          cosh(2r)

julia> op = beamsplitter(b, τ)
GaussianUnitary for 2 modes.
  symplectic basis: QuadBlockBasis
displacement: 4-element Vector{Num}:
 0
 0
 0
 0
symplectic: 4×4 Matrix{Num}:
 sqrt(1 - τ)      sqrt(τ)            0            0
    -sqrt(τ)  sqrt(1 - τ)            0            0
           0            0  sqrt(1 - τ)      sqrt(τ)
           0            0     -sqrt(τ)  sqrt(1 - τ)

julia> newst = ptrace(op * st, 1);
```

Use [Latexify](https://github.com/korsbo/Latexify.jl) to render the covariance matrix of `newst` in LaTeX with the command `latexify(newst.covar) |> print`:

```math
\begin{equation}
\left[
\begin{array}{cc}
\left( \cosh\left( 2 r \right) \sqrt{1 - \tau} + \cos\left( \theta \right) \sinh\left( 2 r \right) \sqrt{\tau} \right) \sqrt{1 - \tau} - \left(  - \sqrt{\tau} \cosh\left( 2 r \right) - \cos\left( \theta \right) \sinh\left( 2 r \right) \sqrt{1 - \tau} \right) \sqrt{\tau} & 2 \sinh\left( 2 r \right) \sin\left( \theta \right) \sqrt{\tau} \sqrt{1 - \tau} \\
2 \sinh\left( 2 r \right) \sin\left( \theta \right) \sqrt{\tau} \sqrt{1 - \tau} & \left( \cosh\left( 2 r \right) \sqrt{1 - \tau} - \cos\left( \theta \right) \sinh\left( 2 r \right) \sqrt{\tau} \right) \sqrt{1 - \tau} - \left(  - \sqrt{\tau} \cosh\left( 2 r \right) + \cos\left( \theta \right) \sinh\left( 2 r \right) \sqrt{1 - \tau} \right) \sqrt{\tau} \\
\end{array}
\right]
\end{equation}
```

## GPU Acceleration

## Multithreading

## Benchmarking and Profiling

