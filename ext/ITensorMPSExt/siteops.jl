##
# Fock-basis (truncated `Qudit`-site) gate operators used to build MPS representations of
# Gaussian states. Ladder operators `"A"`/`"Adag"`/`"N"` are already provided by ITensors'
# built-in `Qudit` site type; here we add the phase-shift, squeeze, displacement, and
# two-mode beam-splitter gates, following Gabs's own unitary conventions in
# `src/unitaries.jl` (`phaseshift`, `squeeze`, `displace`, `beamsplitter`).
##

function _fock_ladder(d::Int)
    a = zeros(ComplexF64, d, d)
    @inbounds for n in 1:d-1
        a[n, n+1] = sqrt(n)
    end
    return a
end

"""
Single-mode phase-shift gate `exp(iθn̂)`, matching the ket-level circuit convention used
throughout this extension (i.e. consistent with [`_reck_decompose`](@ref) and the two-mode
`"BS"` gate below): a passive N-mode circuit built from these gates realizes multiplication
of the single-photon-subspace amplitudes by exactly the target unitary matrix, not its
inverse or conjugate.
"""
function ITensors.op(::OpName"Phase", ::SiteType"Qudit", d::Int; theta::Real)
    return Diagonal(cis.(theta .* (0:d-1)))
end

"""
Single-mode squeezing gate, matching [`Gabs.squeeze`](@ref)'s `(r, theta)` convention:
`exp((theta = 0)) ↦` squeezes the mode's `x`-quadrature for `r > 0`. Exact only in the
`d → ∞` limit; truncation error is largest for large `|r|` and small `d`.
"""
function ITensors.op(::OpName"Squeeze", ::SiteType"Qudit", d::Int; r::Real, theta::Real = 0.0)
    a = _fock_ladder(d)
    adag = a'
    xi = r * cis(theta)
    gen = 0.5 .* (conj(xi) .* (a * a) .- xi .* (adag * adag))
    return exp(Matrix(gen))
end

"""
Single-mode displacement gate `exp(α a† - α* a)`, matching [`Gabs.displace`](@ref)'s `alpha`
convention directly (the Fock-space ladder operators are `ħ`-independent, and Gabs's
`displace` mean-vector convention `√(2ħ)[Re(α), Im(α)]` is defined so that `α` itself needs
no rescaling here). Exact only in the `d → ∞` limit.
"""
function ITensors.op(::OpName"Displace", ::SiteType"Qudit", d::Int; alpha::Number)
    a = _fock_ladder(d)
    adag = a'
    gen = alpha .* adag .- conj(alpha) .* a
    return exp(Matrix(gen))
end

"""
Two-mode beam-splitter gate acting on the truncated `d²`-dimensional two-mode Fock space,
matching the `(mode, θ, φ)` convention produced by [`_reck_decompose`](@ref): the generator
is `θ(e^{iφ} a1†a2 + e^{-iφ} a2†a1)`, and `exp(-i·generator)` induces exactly the target
`[[cosθ, -i sinθ e^{-iφ}], [-i sinθ e^{iφ}, cosθ]]` transform on the single-excitation
subspace `{|1,0⟩, |0,1⟩}` (note the phase sign flip relative to the generator: this
accounts for ITensors' index convention for two-site operator arrays, verified empirically
against ITensors' own built-in two-`Qudit` operators). Dense (`d²×d²`); a banded,
photon-number-conserving implementation is future work. Exact only in the `d → ∞` limit.
"""
function ITensors.op(::OpName"BS", ::SiteType"Qudit", d1::Int, d2::Int; theta::Real, phi::Real)
    d1 == d2 || throw(ArgumentError("Gabs's Fock-basis beam-splitter gate requires equal local dimensions for both modes."))
    a = _fock_ladder(d1)
    Id = Matrix{ComplexF64}(I, d1, d1)
    # site 1 is the *inner* kron factor and site 2 the *outer* one, matching how
    # ITensors reshapes a dense (d1*d2)×(d1*d2) matrix into a (d1,d2,d1,d2) two-site
    # array — verified against ITensors' own built-in "a†b" two-Qudit operator.
    a1 = kron(Id, a)
    a2 = kron(a, Id)
    gen = theta .* (cis(phi) .* (a1' * a2) .+ cis(-phi) .* (a2' * a1))
    U = exp(Matrix(-im .* gen))
    return reshape(U, d1, d2, d1, d2)
end
