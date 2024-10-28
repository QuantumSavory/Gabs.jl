# [Getting Started with Gabs](@id intro)

Gabs.jl is a numerical tooling package for simulating Gaussian quantum information.
Gaussian states and operators have the convenient property that they can be
characterized by low-dimensional matrices in the phase space representation.
Thus, a large class of continuous variable quantum information can be efficiently
simulated on a classical computer, lending to applications in quantum cryptography, quantum machine learning, integrated quantum photonics, and more. Gabs.jl provides a high-level [Julia](https://julialang.org) interface for performing such efficient simulations in a straightforward manner.

In the sections below, a getting started tutorial is provided to introduce the capabilities of Gabs.jl. The rest of the documentation is structured as follows:

- [Manual](@ref) - an overview of the package types and inner-workings,
- [Tutorials](@ref) - explanations for using particular features of the library,
- [Gaussian Zoos](@ref) - a description of predefined Gaussian states and operators,
- [API](@ref Full-API) - the full API of the library,
- [Suggested Readings and References](@ref References) - publications and learning resources on Gaussian quantum information.

!!! note
    This documentation assumes familiarity with linear algebra and quantum information.
    Introductory books and tutorials for these topics are provided in the [Suggested Readings and References page](@ref References).

!!! tip "Contributing"
    Any contribution to Gabs.jl is welcome! Feature suggestions, pull requests, and bug reports
    can be submitted on [Github](https://github.com/apkille/Gabs.jl). The maintainers of Gabs.jl
    subscribe to the [Julia Community Standards](https://julialang.org/community/standards/).

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
